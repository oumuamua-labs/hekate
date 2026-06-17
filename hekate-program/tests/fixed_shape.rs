use hekate_core::trace::ColumnType;
use hekate_math::{Block128, Flat, HardwareField, TowerField};
use hekate_program::{FixedShape, fix, validate_fixed_columns};

type F = Block128;

fn challenges(num_vars: usize) -> Vec<Flat<F>> {
    (0..num_vars)
        .map(|k| F::from(((k as u128) + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)).to_hardware())
        .collect()
}

fn eq(r: &[Flat<F>], index: usize) -> Flat<F> {
    let one = Flat::from_raw(F::ONE);

    let mut prod = one;
    for (k, &r_k) in r.iter().enumerate() {
        prod *= if (index >> k) & 1 == 1 {
            r_k
        } else {
            one - r_k
        };
    }

    prod
}

fn brute_force(values: &[F], r: &[Flat<F>]) -> Flat<F> {
    let mut acc = Flat::from_raw(F::ZERO);
    for (i, &v) in values.iter().enumerate() {
        acc += v.to_hardware() * eq(r, i);
    }

    acc
}

#[test]
fn dense_evaluate_matches_bruteforce() {
    let num_vars = 6;
    let n = 1 << num_vars;
    let r = challenges(num_vars);

    let values: Vec<F> = (0..n).map(|i| F::from((i as u128) * 3 + 1)).collect();

    assert_eq!(
        FixedShape::Dense(values.clone()).evaluate(&r),
        brute_force(&values, &r)
    );
}

#[test]
fn periodic_matches_dense_equivalent() {
    let num_vars = 7;
    let n = 1 << num_vars;
    let r = challenges(num_vars);

    let period = 8;
    let pattern: Vec<F> = (0..period).map(|j| F::from((j as u128) * 5 + 2)).collect();
    let expanded: Vec<F> = (0..n).map(|i| pattern[i % period]).collect();

    let periodic = FixedShape::Periodic {
        period,
        values: pattern,
    };

    assert_eq!(periodic.evaluate(&r), brute_force(&expanded, &r));
}

#[test]
fn sparse_matches_dense_equivalent() {
    let num_vars = 6;
    let n = 1 << num_vars;
    let r = challenges(num_vars);

    let entries = vec![
        (0usize, F::from(7u128)),
        (5, F::from(11u128)),
        (n - 1, F::from(13u128)),
    ];

    let mut expanded = vec![F::ZERO; n];
    for &(row, v) in &entries {
        expanded[row] = v;
    }

    assert_eq!(
        FixedShape::Sparse(entries).evaluate(&r),
        brute_force(&expanded, &r)
    );
}

#[test]
fn single_one_sparse_equals_custom() {
    let num_vars = 6;
    let r = challenges(num_vars);
    let row = 0b101011usize;

    let bits: Vec<bool> = (0..num_vars).map(|k| (row >> k) & 1 == 1).collect();

    let sparse = FixedShape::Sparse(vec![(row, F::ONE)]);
    let custom = FixedShape::<F>::Custom(bits);

    assert_eq!(sparse.evaluate(&r), eq(&r, row));
    assert_eq!(custom.evaluate(&r), eq(&r, row));
    assert_eq!(sparse.evaluate(&r), custom.evaluate(&r));
}

#[test]
fn indicator_shapes_match_kernels() {
    let num_vars = 5;
    let n = 1 << num_vars;
    let r = challenges(num_vars);

    let one = Flat::from_raw(F::ONE);

    // FirstRow is the row-0 indicator;
    // LastRow is the transition mask, i.e.
    // the complement of the row-(n-1) indicator.
    assert_eq!(FixedShape::<F>::FirstRow.evaluate(&r), eq(&r, 0));
    assert_eq!(FixedShape::<F>::LastRow.evaluate(&r), one - eq(&r, n - 1));
}

#[test]
fn validator_accepts_well_formed() {
    let layout = [ColumnType::Bit, ColumnType::B32];
    let fixed = vec![
        fix(0, FixedShape::Dense(vec![F::ONE; 16])),
        fix(1, FixedShape::Sparse(vec![(3, F::from(42u128))])),
    ];

    validate_fixed_columns(&fixed, &layout, Some(4)).expect("well-formed must pass");
}

#[test]
fn validator_rejects_dense_wrong_length() {
    let layout = [ColumnType::B32];
    let fixed = vec![fix(0, FixedShape::Dense(vec![F::ONE; 8]))];

    assert!(validate_fixed_columns(&fixed, &layout, Some(4)).is_err());
}

#[test]
fn validator_rejects_periodic_non_power_of_two() {
    let layout = [ColumnType::B32];
    let fixed = vec![fix(
        0,
        FixedShape::Periodic {
            period: 3,
            values: vec![F::ONE, F::ZERO, F::ONE],
        },
    )];

    assert!(validate_fixed_columns(&fixed, &layout, Some(4)).is_err());
}

#[test]
fn validator_rejects_non_boolean_bit_value() {
    let layout = [ColumnType::Bit];
    let fixed = vec![fix(0, FixedShape::Dense(vec![F::from(2u128); 16]))];

    assert!(validate_fixed_columns(&fixed, &layout, Some(4)).is_err());
}

#[test]
fn validator_rejects_duplicate_pin() {
    let layout = [ColumnType::B32, ColumnType::B32];
    let fixed = vec![
        fix(0, FixedShape::Sparse(vec![(0, F::ONE)])),
        fix(0, FixedShape::Sparse(vec![(1, F::ONE)])),
    ];

    assert!(validate_fixed_columns(&fixed, &layout, Some(4)).is_err());
}

#[test]
fn validator_rejects_out_of_range_col() {
    let layout = [ColumnType::Bit];
    let fixed = vec![fix(5, FixedShape::<F>::LastRow)];

    assert!(validate_fixed_columns(&fixed, &layout, Some(4)).is_err());
}

#[test]
fn validator_rejects_sparse_row_out_of_range() {
    let layout = [ColumnType::B32];
    let fixed = vec![fix(0, FixedShape::Sparse(vec![(99, F::ONE)]))];

    assert!(validate_fixed_columns(&fixed, &layout, Some(4)).is_err());
}

#[test]
fn validator_rejects_duplicate_sparse_row() {
    let layout = [ColumnType::B32];
    let fixed = vec![fix(
        0,
        FixedShape::Sparse(vec![(3, F::ONE), (3, F::from(2u128))]),
    )];

    assert!(validate_fixed_columns(&fixed, &layout, Some(4)).is_err());
}

#[test]
fn value_at_row_matches_evaluate_at_vertices() {
    let num_vars = 5;
    let n = 1 << num_vars;

    let vertex = |row: usize| -> Vec<Flat<F>> {
        (0..num_vars)
            .map(|k| {
                if (row >> k) & 1 == 1 {
                    Flat::from_raw(F::ONE)
                } else {
                    Flat::from_raw(F::ZERO)
                }
            })
            .collect()
    };

    let custom_bits: Vec<bool> = (0..num_vars)
        .map(|k| (0b10110usize >> k) & 1 == 1)
        .collect();

    let shapes = vec![
        FixedShape::<F>::FirstRow,
        FixedShape::<F>::LastRow,
        FixedShape::<F>::Custom(custom_bits),
        FixedShape::Periodic {
            period: 4,
            values: vec![F::ONE, F::ZERO, F::from(5u128), F::ZERO],
        },
        FixedShape::Sparse(vec![(0, F::from(7u128)), (n - 1, F::from(9u128))]),
        FixedShape::Dense((0..n).map(|i| F::from(i as u128)).collect()),
    ];

    for shape in &shapes {
        for row in 0..n {
            assert_eq!(
                shape.value_at_row(row, num_vars),
                shape.evaluate(&vertex(row)),
                "row {row}"
            );
        }
    }
}
