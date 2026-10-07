// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use crate::errors::{Error, Result};
use crate::outer::OuterGeometry;
use crate::proofs::OuterOpening;
use alloc::vec;
use alloc::vec::Vec;
use hekate_crypto::Hasher;
use hekate_crypto::merkle::MerkleTree;
use hekate_math::fft::vanish_eval;
use hekate_math::{
    AdditiveFft, BinaryFieldExtras, CantorBasis, FftError, Flat, HardwareField, TowerField,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

const MAX_LOG: u32 = 63;

/// Mask of a product-code response: `low` covers
/// degrees `< k`, `high` rides `vanisher` for the rest,
/// `Z_k` at each position its consumer reads.
pub struct ProductMask<'a, F> {
    pub low: usize,
    pub high: usize,
    pub vanisher: &'a [Flat<F>],
}

impl<F: TowerField + HardwareField> ProductMask<'_, F> {
    pub fn covers(&self, rows: usize) -> bool {
        self.low < rows && self.high < rows
    }

    pub fn at(&self, position: usize, values: &[Flat<F>]) -> Flat<F> {
        values[self.low] + self.vanisher[position] * values[self.high]
    }
}

pub struct RowEncoder<F> {
    message: AdditiveFft<F>,
    code: AdditiveFft<F>,
    product: AdditiveFft<F>,
    basis: CantorBasis<F>,
    code_shift: Flat<F>,
    code_len: usize,
    domain_len: usize,
}

impl<F: BinaryFieldExtras + HardwareField> RowEncoder<F> {
    pub fn new(geom: &OuterGeometry) -> Result<Self> {
        if !geom.code_len.is_power_of_two() || !geom.domain_len.is_power_of_two() {
            return Err(Error::Protocol {
                protocol: "ligero",
                message: "code and domain lengths must be powers of two",
            });
        }

        let log_k = geom.code_len.ilog2();
        let log_n = geom.domain_len.ilog2();

        if log_k < 1 || log_k >= log_n || log_n > MAX_LOG {
            return Err(Error::Protocol {
                protocol: "ligero",
                message: "require 1 <= log_k < log_n <= 63",
            });
        }

        let basis = CantorBasis::<F>::new(log_n as usize + 1).map_err(|_| Error::Protocol {
            protocol: "ligero",
            message: "Cantor chain has no next basis element",
        })?;

        // beta_{log_n} lies outside W_{log_n}
        let code_shift = basis.betas()[log_n as usize];

        if vanish_eval(log_n as usize, code_shift.to_tower()) == F::ZERO {
            return Err(Error::Protocol {
                protocol: "ligero",
                message: "coset shift landed inside the message subspace",
            });
        }

        let plan = |log: u32| {
            AdditiveFft::new(log).map_err(|e| Error::Protocol {
                protocol: "ligero",
                message: match e {
                    FftError::TwiddleAlloc { .. } => "transform twiddle table allocation failed",
                    _ => "transform size has no Cantor basis",
                },
            })
        };

        Ok(Self {
            message: plan(log_k)?,
            code: plan(log_n)?,
            product: plan(log_k + 1)?,
            basis,
            code_shift,
            code_len: geom.code_len,
            domain_len: geom.domain_len,
        })
    }

    pub fn encode(&self, row: &mut [Flat<F>]) -> Result<()> {
        if row.len() != self.domain_len {
            return Err(Error::Protocol {
                protocol: "ligero",
                message: "row buffer must be domain_len long",
            });
        }

        self.message
            .inverse_scalar(&mut row[..self.code_len])
            .map_err(|_| Error::Protocol {
                protocol: "ligero",
                message: "message interpolation rejected its length",
            })?;

        row[self.code_len..].fill(Flat::from_raw(F::ZERO));

        self.code
            .forward_coset_scalar(row, self.code_shift)
            .map_err(|_| Error::Protocol {
                protocol: "ligero",
                message: "codeword evaluation rejected its length",
            })?;

        Ok(())
    }

    /// Coset-domain values at sorted `columns`
    /// of `2^m` novel-basis coefficients.
    pub fn evaluate_at(&self, coeffs: &[Flat<F>], columns: &[usize]) -> Result<Vec<Flat<F>>> {
        let zero = Flat::from_raw(F::ZERO);

        let mut scratch = vec![zero; coeffs.len().saturating_sub(1)];
        let mut out = vec![zero; columns.len()];

        self.values_at(coeffs, columns, &mut scratch, &mut out)?;

        Ok(out)
    }

    /// `Z_k` at `columns`, as [`Self::message_vanisher`]
    /// holds it: `σ^{log k}` of each column's point.
    pub fn vanisher_at(&self, columns: &[usize]) -> Result<Vec<Flat<F>>> {
        let log_k = self.code_len.trailing_zeros();

        columns
            .iter()
            .map(|&col| {
                let point = match col < self.domain_len {
                    true => self.basis.point(col).ok(),
                    false => None,
                };

                let Some(point) = point else {
                    return Err(Error::Protocol {
                        protocol: "ligero",
                        message: "vanisher column outside the code domain",
                    });
                };

                let mut z = self.code_shift + point;
                for _ in 0..log_k {
                    z = z * z + z;
                }

                Ok(z)
            })
            .collect()
    }

    /// Novel-basis coefficients
    /// of a coset-domain vector.
    pub fn coefficients(&self, values: &[Flat<F>]) -> Result<Vec<Flat<F>>> {
        if values.len() != self.domain_len {
            return Err(Error::Protocol {
                protocol: "ligero",
                message: "coefficient extraction needs a domain_len buffer",
            });
        }

        let mut buf = values.to_vec();

        self.code
            .inverse_coset_scalar(&mut buf, self.code_shift)
            .map_err(|_| Error::Protocol {
                protocol: "ligero",
                message: "coset interpolation rejected its length",
            })?;

        Ok(buf)
    }

    /// Evaluations on `W_{log 2k}` of a
    /// coefficient vector of degree `< 2k`.
    fn product_evaluations(&self, coeffs: &[Flat<F>]) -> Result<Vec<Flat<F>>> {
        let product_len = 2 * self.code_len;

        if coeffs.len() > product_len {
            return Err(Error::Protocol {
                protocol: "ligero",
                message: "product coefficients exceed dimension 2k",
            });
        }

        let mut evals = vec![Flat::from_raw(F::ZERO); product_len];
        evals[..coeffs.len()].copy_from_slice(coeffs);

        self.product
            .forward_scalar(&mut evals)
            .map_err(|_| Error::Protocol {
                protocol: "ligero",
                message: "product evaluation rejected its length",
            })?;

        Ok(evals)
    }

    /// `Z_k` on the code domain.
    pub fn message_vanisher(&self) -> Result<Vec<Flat<F>>> {
        let zero = Flat::from_raw(F::ZERO);
        let mut buf = vec![zero; self.domain_len];

        buf[self.code_len] = Flat::from_raw(F::ONE);

        self.code
            .forward_coset_scalar(&mut buf, self.code_shift)
            .map_err(|_| Error::Protocol {
                protocol: "ligero",
                message: "vanisher evaluation rejected its length",
            })?;

        match self.message_evaluations(&buf)? {
            Some(evals) if evals[..self.code_len].iter().all(|v| *v == zero) => Ok(buf),
            _ => Err(Error::Protocol {
                protocol: "ligero",
                message: "novel basis element k does not vanish on the message domain",
            }),
        }
    }

    pub fn message_evaluations(&self, values: &[Flat<F>]) -> Result<Option<Vec<Flat<F>>>> {
        if values.len() != self.domain_len {
            return Err(Error::Protocol {
                protocol: "ligero",
                message: "response must be a domain_len buffer",
            });
        }

        let mut coeffs = values.to_vec();

        self.code
            .inverse_coset_scalar(&mut coeffs, self.code_shift)
            .map_err(|_| Error::Protocol {
                protocol: "ligero",
                message: "response interpolation rejected its length",
            })?;

        let product_len = 2 * self.code_len;

        // Products of two degree-<k rows and the mask
        // term `Z_k · u` both sit in dimension 2k
        if coeffs[product_len..]
            .iter()
            .any(|c| *c != Flat::from_raw(F::ZERO))
        {
            return Ok(None);
        }

        self.product_evaluations(&coeffs[..product_len]).map(Some)
    }

    pub fn code_len(&self) -> usize {
        self.code_len
    }

    pub fn domain_len(&self) -> usize {
        self.domain_len
    }

    fn values_at(
        &self,
        coeffs: &[Flat<F>],
        columns: &[usize],
        scratch: &mut [Flat<F>],
        out: &mut [Flat<F>],
    ) -> Result<()> {
        if coeffs.len() > self.domain_len || columns.last().is_some_and(|&c| c >= self.domain_len) {
            return Err(Error::Protocol {
                protocol: "ligero",
                message: "coefficients or columns exceed the code domain",
            });
        }

        self.basis
            .evaluate_at(coeffs, self.code_shift, columns, scratch, out)
            .map_err(|_| Error::Protocol {
                protocol: "ligero",
                message: "evaluation needs sorted columns and 2^m coefficients",
            })
    }

    fn encode_at(
        &self,
        message: &[Flat<F>],
        columns: &[usize],
        coeffs: &mut [Flat<F>],
        scratch: &mut [Flat<F>],
    ) -> Result<Vec<Flat<F>>> {
        if message.len() > self.code_len {
            return Err(Error::Protocol {
                protocol: "ligero",
                message: "weight message exceeds the code length",
            });
        }

        let zero = Flat::from_raw(F::ZERO);

        coeffs[..message.len()].copy_from_slice(message);
        coeffs[message.len()..].fill(zero);

        self.message
            .inverse_scalar(coeffs)
            .map_err(|_| Error::Protocol {
                protocol: "ligero",
                message: "message interpolation rejected its length",
            })?;

        let mut out = vec![zero; columns.len()];
        self.values_at(coeffs, columns, scratch, &mut out)?;

        Ok(out)
    }
}

pub struct Opening<F> {
    pub columns: Vec<(usize, Vec<Flat<F>>)>,
    pub siblings: Vec<[u8; 32]>,
}

impl<F: TowerField> Opening<F> {
    pub fn from_wire(wire: &OuterOpening<F>, rows: usize) -> Result<Self>
    where
        F: HardwareField,
    {
        if rows == 0 || wire.values.len() != wire.columns.len() * rows {
            return Err(Error::Protocol {
                protocol: "ligero",
                message: "opening values do not match columns × rows",
            });
        }

        let columns = wire
            .columns
            .iter()
            .zip(wire.values.chunks_exact(rows))
            .map(|(&col, values)| {
                (
                    col as usize,
                    values.iter().map(|v| v.to_hardware()).collect(),
                )
            })
            .collect();

        Ok(Self {
            columns,
            siblings: wire.siblings.clone(),
        })
    }

    pub fn to_wire(&self) -> OuterOpening<F>
    where
        F: HardwareField,
    {
        OuterOpening {
            columns: self.columns.iter().map(|(c, _)| *c as u32).collect(),
            values: self
                .columns
                .iter()
                .flat_map(|(_, v)| v.iter().map(|x| x.to_tower()))
                .collect(),
            siblings: self.siblings.clone(),
        }
    }

    pub fn column_indices(&self) -> Vec<usize> {
        self.columns.iter().map(|(col, _)| *col).collect()
    }

    /// The verifier's view of a stack: the parts' openings
    /// at the same columns, values concatenated per column.
    pub fn stack(parts: &[&Opening<F>]) -> Result<Self> {
        let first = parts.first().ok_or(Error::Protocol {
            protocol: "ligero",
            message: "stack needs at least one opening",
        })?;

        let rows: usize = parts
            .iter()
            .map(|p| p.columns.first().map_or(0, |(_, v)| v.len()))
            .sum();

        let columns = first
            .columns
            .iter()
            .enumerate()
            .map(|(i, (col, _))| {
                let mut values = Vec::with_capacity(rows);
                for part in parts {
                    match part.columns.get(i) {
                        Some((c, v)) if c == col => values.extend_from_slice(v),
                        _ => {
                            return Err(Error::Protocol {
                                protocol: "ligero",
                                message: "stacked openings must cover the same columns",
                            });
                        }
                    }
                }

                Ok((*col, values))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            columns,
            siblings: Vec::new(),
        })
    }
}

/// Rows are indexed by opening position, not by column.
pub struct OpenedWeights<F> {
    columns: Vec<usize>,
    rows: Vec<Vec<Flat<F>>>,
}

impl<F> OpenedWeights<F> {
    fn covers(&self, opening: &Opening<F>) -> bool {
        self.columns.len() == opening.columns.len()
            && self
                .columns
                .iter()
                .zip(&opening.columns)
                .all(|(&want, (col, _))| want == *col)
            && self.rows.iter().all(|r| r.len() == self.columns.len())
    }
}

/// The response's degree bound is `coeffs.len()`.
pub fn verify_interleaved<F: BinaryFieldExtras + HardwareField + TowerField>(
    encoder: &RowEncoder<F>,
    coeffs: &[Flat<F>],
    r_int: &[Flat<F>],
    mask: usize,
    opening: &Opening<F>,
) -> bool {
    if coeffs.len() != encoder.code_len() {
        return false;
    }

    let Ok(response) = encoder.evaluate_at(coeffs, &opening.column_indices()) else {
        return false;
    };

    opening
        .columns
        .iter()
        .zip(&response)
        .all(|((_, values), &expected)| {
            if mask >= values.len() || r_int.len() != values.len() - 1 {
                return false;
            }

            let mut acc = values[mask];
            let mut next = 0;

            for (r, v) in values.iter().enumerate() {
                if r == mask {
                    continue;
                }

                acc += r_int[next] * *v;
                next += 1;
            }

            acc == expected
        })
}

pub fn verify_linear<F: BinaryFieldExtras + HardwareField + TowerField>(
    encoder: &RowEncoder<F>,
    coeffs: &[Flat<F>],
    weights: &OpenedWeights<F>,
    rows_used: &[usize],
    mask: &ProductMask<'_, F>,
    target: Flat<F>,
    opening: &Opening<F>,
) -> bool {
    if coeffs.len() != 2 * encoder.code_len()
        || mask.vanisher.len() != opening.columns.len()
        || weights.rows.len() != rows_used.len()
        || !weights.covers(opening)
    {
        return false;
    }

    let Ok(evals) = encoder.product_evaluations(coeffs) else {
        return false;
    };

    let sum = evals[..encoder.code_len()]
        .iter()
        .fold(Flat::from_raw(F::ZERO), |a, b| a + *b);

    if sum != target {
        return false;
    }

    let Ok(response) = encoder.evaluate_at(coeffs, &opening.column_indices()) else {
        return false;
    };

    opening
        .columns
        .iter()
        .zip(&response)
        .enumerate()
        .all(|(i, ((_, values), &expected))| {
            if !mask.covers(values.len()) || rows_used.iter().any(|&r| r >= values.len()) {
                return false;
            }

            let mut acc = mask.at(i, values);
            for (weight_row, &row) in weights.rows.iter().zip(rows_used) {
                acc += weight_row[i] * values[row];
            }

            acc == expected
        })
}

pub fn verify_quadratic<F: BinaryFieldExtras + HardwareField + TowerField>(
    encoder: &RowEncoder<F>,
    coeffs: &[Flat<F>],
    triples: &[[usize; 3]],
    r_quad: &[Flat<F>],
    mask: &ProductMask<'_, F>,
    message_len: usize,
    opening: &Opening<F>,
) -> bool {
    if coeffs.len() != 2 * encoder.code_len()
        || mask.vanisher.len() != opening.columns.len()
        || r_quad.len() != triples.len()
    {
        return false;
    }

    let Ok(evals) = encoder.product_evaluations(coeffs) else {
        return false;
    };

    if evals[..message_len]
        .iter()
        .any(|v| *v != Flat::from_raw(F::ZERO))
    {
        return false;
    }

    let Ok(response) = encoder.evaluate_at(coeffs, &opening.column_indices()) else {
        return false;
    };

    opening
        .columns
        .iter()
        .zip(&response)
        .enumerate()
        .all(|(i, ((_, values), &expected))| {
            if !mask.covers(values.len()) || triples.iter().flatten().any(|&r| r >= values.len()) {
                return false;
            }

            let mut acc = mask.at(i, values);
            for (t, [x, y, z]) in triples.iter().enumerate() {
                acc += r_quad[t] * (values[*x] * values[*y] - values[*z]);
            }

            acc == expected
        })
}

/// Checks the wire opening against `root`;
/// leaves are hashed from the wire's
/// tower bytes, as [`column_leaf`] does.
pub fn verify_opening<F: TowerField + HardwareField, H: Hasher>(
    root: &[u8; 32],
    domain_len: usize,
    rows: usize,
    opening: &OuterOpening<F>,
) -> bool {
    if rows == 0 || opening.values.len() != opening.columns.len() * rows {
        return false;
    }

    let leaves: Vec<(usize, [u8; 32])> = opening
        .columns
        .iter()
        .zip(opening.values.chunks_exact(rows))
        .map(|(&col, values)| (col as usize, column_leaf::<F, H>(values.iter().copied())))
        .collect();

    MerkleTree::<F, H>::verify_batch(root, domain_len, &leaves, &opening.siblings)
}

pub fn weights_at_columns<F: BinaryFieldExtras + HardwareField + TowerField>(
    encoder: &RowEncoder<F>,
    messages: &[Vec<Flat<F>>],
    columns: &[usize],
) -> Result<OpenedWeights<F>> {
    if columns.iter().any(|&c| c >= encoder.domain_len()) {
        return Err(Error::Protocol {
            protocol: "ligero",
            message: "weight column out of range",
        });
    }

    let zero = Flat::from_raw(F::ZERO);
    let k = encoder.code_len();
    let buffers = || (vec![zero; k], vec![zero; k - 1]);

    #[cfg(feature = "parallel")]
    let rows = messages
        .par_iter()
        .map_init(buffers, |(coeffs, scratch), message| {
            encoder.encode_at(message, columns, coeffs, scratch)
        })
        .collect::<Result<Vec<_>>>()?;

    #[cfg(not(feature = "parallel"))]
    let rows = {
        let (mut coeffs, mut scratch) = buffers();

        messages
            .iter()
            .map(|message| encoder.encode_at(message, columns, &mut coeffs, &mut scratch))
            .collect::<Result<Vec<_>>>()?
    };

    Ok(OpenedWeights {
        columns: columns.to_vec(),
        rows,
    })
}

/// Leaf of one column, its rows top to bottom.
pub fn column_leaf<F: TowerField, H: Hasher>(column: impl Iterator<Item = F>) -> [u8; 32] {
    let mut hasher = H::new();
    hasher.update(&[0u8]);
    hasher.update_fields(&[], column);

    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use alloc::vec;
    use alloc::vec::Vec;
    use hekate_math::{Block128, TowerField};

    const OPENED: [usize; 4] = [1, 9, 40, 77];

    type F = Block128;

    fn geometry() -> OuterGeometry {
        Config::prod().outer_geom(4_075, 2_384, 128).unwrap()
    }

    fn geometries() -> Vec<OuterGeometry> {
        let config = Config::prod();

        [(1, 0), (4_075, 2_384), (41_025, 34_648)]
            .iter()
            .map(|&(scalars, wires)| config.outer_geom(scalars, wires, 128).unwrap())
            .collect()
    }

    fn mix(seed: u128) -> Flat<F> {
        F::from(
            seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(0x51ed_2701),
        )
        .to_hardware()
    }

    fn message(geom: &OuterGeometry, salt: u128) -> Vec<Flat<F>> {
        let mut row = vec![Flat::from_raw(F::ZERO); geom.domain_len];
        for (i, slot) in row[..geom.code_len].iter_mut().enumerate() {
            *slot = mix(i as u128 + salt);
        }

        row
    }

    fn opening_at(rows: &[Vec<Flat<F>>], columns: &[usize]) -> Opening<F> {
        Opening {
            columns: columns
                .iter()
                .map(|&c| (c, rows.iter().map(|r| r[c]).collect()))
                .collect(),
            siblings: Vec::new(),
        }
    }

    fn encoded(encoder: &RowEncoder<F>, geom: &OuterGeometry, salt: u128) -> Vec<Flat<F>> {
        let mut row = message(geom, salt);
        encoder.encode(&mut row).unwrap();

        row
    }

    /// Uniform message summing to zero over the code domain.
    fn zero_sum_row(encoder: &RowEncoder<F>, geom: &OuterGeometry, salt: u128) -> Vec<Flat<F>> {
        let mut row = vec![Flat::from_raw(F::ZERO); geom.domain_len];
        let mut acc = Flat::from_raw(F::ZERO);

        for (i, slot) in row[..geom.code_len - 1].iter_mut().enumerate() {
            *slot = mix(i as u128 + salt);
            acc += *slot;
        }

        row[geom.code_len - 1] = acc;
        encoder.encode(&mut row).unwrap();

        row
    }

    /// Hull-Dobell: `37x + 11` has full
    /// period on a power-of-two modulus.
    fn spread_columns(geom: &OuterGeometry) -> Vec<usize> {
        let mut columns = Vec::with_capacity(geom.queries);
        let mut c = 0usize;

        while columns.len() < geom.queries {
            c = (c * 37 + 11) % geom.domain_len;

            if !columns.contains(&c) {
                columns.push(c);
            }
        }

        columns
    }

    fn rank(mut rows: Vec<Vec<Flat<F>>>) -> usize {
        let zero = Flat::from_raw(F::ZERO);
        let cols = rows.first().map_or(0, Vec::len);

        let mut rank = 0;
        for col in 0..cols {
            let Some(pivot) = (rank..rows.len()).find(|&r| rows[r][col] != zero) else {
                continue;
            };

            rows.swap(rank, pivot);

            let inv = rows[rank][col].to_tower().invert().to_hardware();
            for v in rows[rank][col..].iter_mut() {
                *v *= inv;
            }

            let pivot_row = rows[rank].clone();

            for (r, row) in rows.iter_mut().enumerate() {
                if r == rank || row[col] == zero {
                    continue;
                }

                let factor = row[col];
                for (v, p) in row[col..].iter_mut().zip(&pivot_row[col..]) {
                    *v += *p * factor;
                }
            }

            rank += 1;
        }

        rank
    }

    /// `Z` vanishes on the message subspace:
    /// `low` is the response there.
    fn split_product_code(
        encoder: &RowEncoder<F>,
        geom: &OuterGeometry,
        vanisher: &[Flat<F>],
        response: &[Flat<F>],
    ) -> (Vec<Flat<F>>, Vec<Flat<F>>) {
        let evals = encoder.message_evaluations(response).unwrap().unwrap();

        let mut low = vec![Flat::from_raw(F::ZERO); geom.domain_len];
        low[..geom.code_len].copy_from_slice(&evals[..geom.code_len]);

        encoder.encode(&mut low).unwrap();

        let quotient: Vec<Flat<F>> = (0..geom.domain_len)
            .map(|c| (response[c] + low[c]) * vanisher[c].to_tower().invert().to_hardware())
            .collect();

        let high_evals = encoder.message_evaluations(&quotient).unwrap().unwrap();

        let mut high = vec![Flat::from_raw(F::ZERO); geom.domain_len];
        high[..geom.code_len].copy_from_slice(&high_evals[..geom.code_len]);

        encoder.encode(&mut high).unwrap();

        (low, high)
    }

    fn wire_form(encoder: &RowEncoder<F>, response: &[Flat<F>], len: usize) -> Vec<Flat<F>> {
        encoder.coefficients(response).unwrap()[..len].to_vec()
    }

    fn interleaved_response(
        rows: &[Vec<Flat<F>>],
        coeffs: &[Flat<F>],
        mask: usize,
        domain_len: usize,
    ) -> Vec<Flat<F>> {
        (0..domain_len)
            .map(|c| {
                let mut acc = rows[mask][c];
                for (r, k) in coeffs.iter().enumerate() {
                    acc += *k * rows[r][c];
                }

                acc
            })
            .collect()
    }

    #[test]
    fn code_domain_is_disjoint_from_message_domain() {
        let geom = geometry();
        let encoder = RowEncoder::<F>::new(&geom).unwrap();
        let log_n = geom.domain_len.ilog2() as usize;

        assert_ne!(vanish_eval(log_n, encoder.code_shift.to_tower()), F::ZERO);
    }

    #[test]
    fn encoding_is_linear() {
        let geom = geometry();
        let encoder = RowEncoder::<F>::new(&geom).unwrap();

        let mut a = message(&geom, 1);
        let mut b = message(&geom, 5_000);
        let mut sum: Vec<Flat<F>> = a.iter().zip(&b).map(|(x, y)| *x + *y).collect();

        encoder.encode(&mut a).unwrap();
        encoder.encode(&mut b).unwrap();
        encoder.encode(&mut sum).unwrap();

        for i in 0..geom.domain_len {
            assert_eq!(a[i] + b[i], sum[i]);
        }
    }

    #[test]
    fn zero_message_encodes_to_zero_codeword() {
        let geom = geometry();
        let encoder = RowEncoder::<F>::new(&geom).unwrap();
        let mut row = vec![Flat::from_raw(F::ZERO); geom.domain_len];

        encoder.encode(&mut row).unwrap();

        assert!(row.iter().all(|v| *v == Flat::from_raw(F::ZERO)));
    }

    #[test]
    fn encoding_agrees_with_coefficient_path() {
        let geom = geometry();
        let encoder = RowEncoder::<F>::new(&geom).unwrap();

        let coeffs: Vec<Flat<F>> = (0..geom.code_len).map(|i| mix(i as u128 + 77)).collect();

        let mut evals = vec![Flat::from_raw(F::ZERO); geom.domain_len];
        evals[..geom.code_len].copy_from_slice(&coeffs);

        AdditiveFft::<F>::new(geom.code_len.ilog2())
            .unwrap()
            .forward_scalar(&mut evals[..geom.code_len])
            .unwrap();

        let mut direct = vec![Flat::from_raw(F::ZERO); geom.domain_len];
        direct[..geom.code_len].copy_from_slice(&coeffs);

        AdditiveFft::<F>::new(geom.domain_len.ilog2())
            .unwrap()
            .forward_coset_scalar(&mut direct, encoder.code_shift)
            .unwrap();

        encoder.encode(&mut evals).unwrap();

        assert_eq!(evals, direct);
    }

    /// The bijection behind the length gate:
    /// it accepts what `is_codeword` used to accept.
    #[test]
    fn coefficients_invert_evaluate_at_both_degree_bounds() {
        let geom = geometry();
        let encoder = RowEncoder::<F>::new(&geom).unwrap();

        let all: Vec<usize> = (0..geom.domain_len).collect();

        for len in [geom.code_len, 2 * geom.code_len] {
            let coeffs: Vec<Flat<F>> = (0..len).map(|i| mix((i + len) as u128 + 5)).collect();
            let codeword = encoder.evaluate_at(&coeffs, &all).unwrap();
            let back = encoder.coefficients(&codeword).unwrap();

            assert_eq!(codeword.len(), geom.domain_len, "{len}");
            assert_eq!(back[..len], coeffs[..], "{len}");
            assert!(
                back[len..].iter().all(|c| *c == Flat::from_raw(F::ZERO)),
                "{len}"
            );
        }
    }

    #[test]
    fn nonzero_message_is_far_from_zero() {
        let geom = geometry();
        let encoder = RowEncoder::<F>::new(&geom).unwrap();
        let mut row = message(&geom, 31);

        encoder.encode(&mut row).unwrap();

        let zeros = row
            .iter()
            .filter(|v| **v == Flat::from_raw(F::ZERO))
            .count();

        assert!(zeros < geom.code_len);
    }

    #[test]
    fn wrong_length_buffer_is_rejected() {
        let geom = geometry();
        let encoder = RowEncoder::<F>::new(&geom).unwrap();
        let mut row = vec![Flat::from_raw(F::ZERO); geom.domain_len - 1];

        assert!(encoder.encode(&mut row).is_err());
    }

    /// `coeffs.len()` is the only degree gate;
    /// a codeword is all the wire can express.
    #[test]
    fn interleaved_test_rejects_wrong_degree_bound() {
        let geom = geometry();
        let encoder = RowEncoder::<F>::new(&geom).unwrap();

        let rows: Vec<Vec<Flat<F>>> = (0..4)
            .map(|i| encoded(&encoder, &geom, 100 * i + 1))
            .collect();

        let coeffs: Vec<Flat<F>> = (0..3).map(|i| mix(i as u128 + 7_000)).collect();
        let mask = 3;

        let honest = interleaved_response(&rows, &coeffs, mask, geom.domain_len);
        let wire = wire_form(&encoder, &honest, geom.code_len);
        let opening = opening_at(&rows, &OPENED);

        assert!(verify_interleaved(&encoder, &wire, &coeffs, mask, &opening));

        let mut long = wire.clone();
        long.push(Flat::from_raw(F::ONE));

        assert!(!verify_interleaved(
            &encoder, &long, &coeffs, mask, &opening
        ));

        assert!(!verify_interleaved(
            &encoder,
            &wire[..wire.len() - 1],
            &coeffs,
            mask,
            &opening
        ));
    }

    #[test]
    fn interleaved_test_rejects_column_that_misses_response() {
        let geom = geometry();
        let encoder = RowEncoder::<F>::new(&geom).unwrap();

        let rows: Vec<Vec<Flat<F>>> = (0..4)
            .map(|i| encoded(&encoder, &geom, 200 * i + 3))
            .collect();

        let coeffs: Vec<Flat<F>> = (0..3).map(|i| mix(i as u128 + 8_000)).collect();
        let mask = 3;

        let response = interleaved_response(&rows, &coeffs, mask, geom.domain_len);
        let mut opening = opening_at(&rows, &OPENED);

        opening.columns[0].1[0] += Flat::from_raw(F::ONE);

        assert!(!verify_interleaved(
            &encoder,
            &wire_form(&encoder, &response, geom.code_len),
            &coeffs,
            mask,
            &opening,
        ));
    }

    #[test]
    fn linear_test_rejects_wrong_target() {
        let geom = geometry();
        let encoder = RowEncoder::<F>::new(&geom).unwrap();
        let vanisher = encoder.message_vanisher().unwrap();

        let data_msgs: Vec<Vec<Flat<F>>> = (0..2).map(|i| message(&geom, 300 * i + 11)).collect();
        let weight_msgs: Vec<Vec<Flat<F>>> = (0..2)
            .map(|i| {
                (0..geom.message_len)
                    .map(|c| mix(c as u128 + 400 * i + 17))
                    .collect()
            })
            .collect();

        let mut target = Flat::from_raw(F::ZERO);
        for (w, u) in weight_msgs.iter().zip(&data_msgs) {
            for c in 0..geom.message_len {
                target += w[c] * u[c];
            }
        }

        let mut rows: Vec<Vec<Flat<F>>> = data_msgs
            .iter()
            .map(|m| {
                let mut row = m.clone();
                encoder.encode(&mut row).unwrap();

                row
            })
            .collect();

        rows.push(zero_sum_row(&encoder, &geom, 991));
        rows.push(encoded(&encoder, &geom, 1_313));

        let weights: Vec<Vec<Flat<F>>> = weight_msgs
            .iter()
            .map(|m| {
                let mut row = vec![Flat::from_raw(F::ZERO); geom.domain_len];
                row[..m.len()].copy_from_slice(m);

                encoder.encode(&mut row).unwrap();

                row
            })
            .collect();

        let opened_weights = weights_at_columns(&encoder, &weight_msgs, &OPENED).unwrap();
        let opened_vanisher = encoder.vanisher_at(&OPENED).unwrap();

        let mask = ProductMask {
            low: 2,
            high: 3,
            vanisher: &opened_vanisher,
        };

        let response: Vec<Flat<F>> = (0..geom.domain_len)
            .map(|c| {
                let mut acc = rows[2][c] + vanisher[c] * rows[3][c];
                for (w, u) in weights.iter().zip(&rows) {
                    acc += w[c] * u[c];
                }

                acc
            })
            .collect();

        let rows_used = [0usize, 1];
        let opening = opening_at(&rows, &OPENED);
        let wire = wire_form(&encoder, &response, 2 * geom.code_len);

        assert!(verify_linear(
            &encoder,
            &wire,
            &opened_weights,
            &rows_used,
            &mask,
            target,
            &opening,
        ));

        assert!(!verify_linear(
            &encoder,
            &wire,
            &opened_weights,
            &rows_used,
            &mask,
            target + Flat::from_raw(F::ONE),
            &opening,
        ));
    }

    #[test]
    fn quadratic_test_rejects_broken_triple() {
        let geom = geometry();
        let encoder = RowEncoder::<F>::new(&geom).unwrap();
        let vanisher = encoder.message_vanisher().unwrap();

        let build = |break_triple: bool| -> (Vec<Vec<Flat<F>>>, Vec<Flat<F>>) {
            let x = message(&geom, 501);
            let y = message(&geom, 607);

            let mut z = message(&geom, 709);
            for c in 0..geom.message_len {
                z[c] = x[c] * y[c];
            }

            if break_triple {
                z[0] += Flat::from_raw(F::ONE);
            }

            let mut rows: Vec<Vec<Flat<F>>> = [x, y, z]
                .into_iter()
                .map(|mut m| {
                    encoder.encode(&mut m).unwrap();

                    m
                })
                .collect();

            let mut low = vec![Flat::from_raw(F::ZERO); geom.domain_len];
            for (c, slot) in low[geom.message_len..geom.code_len].iter_mut().enumerate() {
                *slot = mix(c as u128 + 811);
            }

            encoder.encode(&mut low).unwrap();

            rows.push(low);
            rows.push(encoded(&encoder, &geom, 907));

            let r_quad = [mix(1_009)];
            let response: Vec<Flat<F>> = (0..geom.domain_len)
                .map(|c| {
                    let mut acc = rows[3][c] + vanisher[c] * rows[4][c];
                    acc += r_quad[0] * (rows[0][c] * rows[1][c] - rows[2][c]);

                    acc
                })
                .collect();

            (rows, response)
        };

        let opened_vanisher = encoder.vanisher_at(&OPENED).unwrap();

        let mask = ProductMask {
            low: 3,
            high: 4,
            vanisher: &opened_vanisher,
        };

        let triples = [[0usize, 1, 2]];
        let r_quad = [mix(1_009)];

        let (rows, response) = build(false);

        assert!(verify_quadratic(
            &encoder,
            &wire_form(&encoder, &response, 2 * geom.code_len),
            &triples,
            &r_quad,
            &mask,
            geom.message_len,
            &opening_at(&rows, &OPENED),
        ));

        let (rows, response) = build(true);

        assert!(!verify_quadratic(
            &encoder,
            &wire_form(&encoder, &response, 2 * geom.code_len),
            &triples,
            &r_quad,
            &mask,
            geom.message_len,
            &opening_at(&rows, &OPENED),
        ));
    }

    /// Row privacy: an opening is independent of
    /// the row's message (Ligero 2017 Lemma 4.15).
    #[test]
    fn filler_reaches_every_opening_pattern() {
        let geom = geometry();
        let encoder = RowEncoder::<F>::new(&geom).unwrap();

        let units: Vec<Vec<Flat<F>>> = (geom.message_len..geom.code_len)
            .map(|i| {
                let mut row = vec![Flat::from_raw(F::ZERO); geom.domain_len];
                row[i] = Flat::from_raw(F::ONE);

                encoder.encode(&mut row).unwrap();

                row
            })
            .collect();

        assert_eq!(units.len(), geom.queries + 1);

        for columns in [spread_columns(&geom), (0..geom.queries).collect()] {
            let map: Vec<Vec<Flat<F>>> = units
                .iter()
                .map(|row| columns.iter().map(|&c| row[c]).collect())
                .collect();

            assert_eq!(rank(map), geom.queries);
        }
    }

    /// Injective on `RS[n,k]^2`, hence onto all the
    /// product code: the mask reaches the high half.
    #[test]
    fn product_code_split_is_unique() {
        let geom = geometry();
        let encoder = RowEncoder::<F>::new(&geom).unwrap();
        let vanisher = encoder.message_vanisher().unwrap();

        assert!(vanisher.iter().all(|v| *v != Flat::from_raw(F::ZERO)));

        let zero = vec![Flat::from_raw(F::ZERO); geom.domain_len];
        let low = encoded(&encoder, &geom, 4_201);
        let high = encoded(&encoder, &geom, 9_907);

        for (a, b) in [(&low, &high), (&low, &zero), (&zero, &high)] {
            let response: Vec<Flat<F>> = (0..geom.domain_len)
                .map(|c| a[c] + vanisher[c] * b[c])
                .collect();

            assert!(encoder.message_evaluations(&response).unwrap().is_some());

            let (split_low, split_high) = split_product_code(&encoder, &geom, &vanisher, &response);

            assert_eq!(&split_low, a);
            assert_eq!(&split_high, b);
        }
    }

    #[test]
    fn evaluate_at_matches_full_coset_transform() {
        for geom in geometries() {
            let encoder = RowEncoder::<F>::new(&geom).unwrap();
            let code = AdditiveFft::<F>::new(geom.domain_len.ilog2()).unwrap();

            let mut spread = spread_columns(&geom);
            spread.sort_unstable();

            let all: Vec<usize> = (0..geom.domain_len).collect();
            let edges = vec![0, geom.domain_len - 1];

            for len in [geom.code_len, 2 * geom.code_len] {
                let coeffs: Vec<Flat<F>> = (0..len).map(|i| mix((i + 3 * len) as u128)).collect();

                let mut full = vec![Flat::from_raw(F::ZERO); geom.domain_len];
                full[..len].copy_from_slice(&coeffs);

                code.forward_coset_scalar(&mut full, encoder.code_shift)
                    .unwrap();

                for columns in [&OPENED.to_vec(), &spread, &edges, &all] {
                    let want: Vec<Flat<F>> = columns.iter().map(|&c| full[c]).collect();

                    assert_eq!(
                        encoder.evaluate_at(&coeffs, columns).unwrap(),
                        want,
                        "n {} len {len}",
                        geom.domain_len
                    );
                }
            }
        }
    }

    #[test]
    fn weights_at_columns_match_encoded_rows() {
        let geom = geometry();
        let encoder = RowEncoder::<F>::new(&geom).unwrap();

        let messages: Vec<Vec<Flat<F>>> = [geom.message_len, geom.code_len, 1]
            .iter()
            .enumerate()
            .map(|(r, &len)| (0..len).map(|c| mix((c + 7_000 * r) as u128)).collect())
            .collect();

        let mut spread = spread_columns(&geom);
        spread.sort_unstable();

        let opened = weights_at_columns(&encoder, &messages, &spread).unwrap();

        for (message, row) in messages.iter().zip(&opened.rows) {
            let mut full = vec![Flat::from_raw(F::ZERO); geom.domain_len];
            full[..message.len()].copy_from_slice(message);

            encoder.encode(&mut full).unwrap();

            let want: Vec<Flat<F>> = spread.iter().map(|&c| full[c]).collect();

            assert_eq!(*row, want);
        }
    }

    #[test]
    fn vanisher_at_matches_message_vanisher() {
        for geom in geometries() {
            let encoder = RowEncoder::<F>::new(&geom).unwrap();

            let all: Vec<usize> = (0..geom.domain_len).collect();

            assert_eq!(
                encoder.vanisher_at(&all).unwrap(),
                encoder.message_vanisher().unwrap(),
                "n {}",
                geom.domain_len
            );
        }
    }

    #[test]
    fn evaluation_rejects_unsorted_or_outside_columns() {
        let geom = geometry();
        let encoder = RowEncoder::<F>::new(&geom).unwrap();
        let coeffs: Vec<Flat<F>> = (0..geom.code_len).map(|i| mix(i as u128)).collect();

        assert!(encoder.evaluate_at(&coeffs, &[9, 1]).is_err());
        assert!(encoder.evaluate_at(&coeffs, &[1, geom.domain_len]).is_err());
        assert!(encoder.vanisher_at(&[geom.domain_len]).is_err());
        assert!(weights_at_columns(&encoder, core::slice::from_ref(&coeffs), &[40, 9]).is_err());
    }
}
