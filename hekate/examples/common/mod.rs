// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::proofs::InnerProof;
use hekate_math::TowerField;
use hekate_program::Program;
use hekate_program::digest::{program_id, program_id_hex};
use std::time::Instant;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::FmtSpan;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

pub fn init(name: &str) {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug")),
        )
        .with_span_events(FmtSpan::CLOSE)
        .try_init();

    hekate_prover_sys::init_tracing();

    println!("==================================================");
    println!("Hekate: {}", name);
}

pub fn phase<T>(label: &str, f: impl FnOnce() -> T) -> T {
    println!("\n-> {}...\n", label);

    let start = Instant::now();
    let result = f();

    println!("\n-> {} done in {:.2?}\n", label, start.elapsed());

    result
}

/// Times a labelled phase; under `dhat-heap` also
/// reports its isolated heap footprint. The dhat
/// profiler is a per-run singleton, wrap exactly one phase.
pub fn phase_with_mem<T>(label: &str, f: impl FnOnce() -> T) -> T {
    #[cfg(feature = "dhat-heap")]
    {
        let _profiler = dhat::Profiler::builder().testing().build();
        let start = dhat::HeapStats::get();

        println!("\n-> {}...\n", label);

        let t = Instant::now();
        let result = f();
        let elapsed = t.elapsed();

        let end = dhat::HeapStats::get();

        println!("\n-> {} done in {:.2?}\n", label, elapsed);

        report_phase_mem(label, &start, &end);

        result
    }

    #[cfg(not(feature = "dhat-heap"))]
    {
        phase(label, f)
    }
}

#[cfg(feature = "dhat-heap")]
fn report_phase_mem(label: &str, start: &dhat::HeapStats, end: &dhat::HeapStats) {
    println!("--------------------------------------------------");
    println!("  {} MEMORY (dhat, this phase only)", label);
    println!("--------------------------------------------------");
    println!(
        "  Peak live heap:   {:>10.2} KB",
        end.max_bytes as f64 / 1024.0
    );
    println!(
        "  Total allocated:  {:>10.2} KB",
        (end.total_bytes - start.total_bytes) as f64 / 1024.0
    );
    println!(
        "  Alloc count:      {:>10}",
        end.total_blocks - start.total_blocks
    );
    println!(
        "  Live at end:      {:>10.2} KB",
        end.curr_bytes as f64 / 1024.0
    );
    println!("--------------------------------------------------");
}

/// Auto-detects chiplet and LogUp bus sections.
pub fn proof_breakdown<F>(proof: &InnerProof<F>)
where
    F: TowerField + serde::Serialize,
{
    let bin_cfg = bincode::config::standard();

    let total_bytes =
        bincode::serde::encode_to_vec(proof, bin_cfg).expect("Proof serialization failed");
    let total_sz = total_bytes.len();

    let wire_sz = hekate_sdk::serialize_proof_bytes(proof).len();

    println!(
        "   Total Proof size: {:.2} KB ({} bytes)\n",
        wire_sz as f64 / 1024.0,
        wire_sz
    );

    // Main components
    let trace_comm_sz = enc_size(&proof.trace_commitment, bin_cfg);
    let zcheck_sz = enc_size(&proof.zerocheck_proof, bin_cfg);

    // Eval batch argument breakdown
    let eval_sc_sz = enc_size(&proof.eval_proof.sumcheck_proof, bin_cfg);
    let eval_tensor_sz = enc_size(&proof.eval_proof.tensor_vec, bin_cfg);
    let eval_master_sz = enc_size(&proof.eval_proof.master_evals, bin_cfg);
    let eval_pt_sz = enc_size(&proof.eval_proof.point_evaluation, bin_cfg);
    let ldt_batch_sz = enc_size(&proof.eval_proof.ldt_proof.batch_path, bin_cfg);
    let ldt_opened_sz = enc_size(&proof.eval_proof.ldt_proof.opened_columns, bin_cfg);
    let eval_h_sz = enc_size(&proof.eval_proof.h_ldt_proof, bin_cfg);

    println!("--------------------------------------------------");
    println!("  PROOF COMPONENT BREAKDOWN (bincode)");
    println!("--------------------------------------------------");
    println!("  Trace Commitment:       {:>8} bytes", trace_comm_sz);
    println!("  Main AIR ZeroCheck:     {:>8} bytes", zcheck_sz);
    println!("  Eval Batch Argument:");
    println!("    Eval Sumcheck:        {:>8} bytes", eval_sc_sz);
    println!("    Tensor Vector (q):    {:>8} bytes", eval_tensor_sz);
    println!("    Master Evals:         {:>8} bytes", eval_master_sz);
    println!("    Point Evaluation:     {:>8} bytes", eval_pt_sz);
    println!("    LDT Batch Path:       {:>8} bytes", ldt_batch_sz);
    println!("    LDT Opened Columns:   {:>8} bytes", ldt_opened_sz);
    println!("    H Opening:            {:>8} bytes", eval_h_sz);

    if !proof.chiplet_commitments.is_empty() {
        let n = proof.chiplet_commitments.len();
        let chip_comm_sz = enc_size(&proof.chiplet_commitments, bin_cfg);
        let chip_zc_sz = enc_size(&proof.chiplet_zerocheck_proofs, bin_cfg);
        let chip_eval_sz = enc_size(&proof.chiplet_eval_proofs, bin_cfg);
        let chip_h_sz = proof
            .chiplet_eval_proofs
            .iter()
            .map(|p| enc_size(&p.h_ldt_proof, bin_cfg))
            .sum::<usize>();

        println!("  Chiplets ({}):", n);
        println!("    Commitments:          {:>8} bytes", chip_comm_sz);
        println!("    ZeroChecks:           {:>8} bytes", chip_zc_sz);
        println!("    Eval Arguments:       {:>8} bytes", chip_eval_sz);
        println!("      of which h openings:{:>8} bytes", chip_h_sz);
    }

    let main_bus_count = proof.main_logup_aux.h_evals.len();
    let chip_bus_count: usize = proof
        .chiplet_logup_aux
        .iter()
        .map(|a| a.h_evals.len())
        .sum();

    if main_bus_count + chip_bus_count > 0 {
        let main_logup_sz = enc_size(&proof.main_logup_aux, bin_cfg);
        let chip_logup_sz = enc_size(&proof.chiplet_logup_aux, bin_cfg);

        println!(
            "  LogUp Bus Aux ({} specs):",
            main_bus_count + chip_bus_count
        );
        println!("    Main:                 {:>8} bytes", main_logup_sz);
        println!("    Chiplets:             {:>8} bytes", chip_logup_sz);
    }

    let pad_root_sz = enc_size(&proof.pad_root, bin_cfg);

    if proof.pad_root.is_some() {
        println!("  PAD Root:               {:>8} bytes", pad_root_sz);
    }

    let outer_sz = outer_breakdown(proof, bin_cfg);

    let itemized = enc_size(&proof.trace_commitment, bin_cfg)
        + enc_size(&proof.zerocheck_proof, bin_cfg)
        + enc_size(&proof.main_logup_aux, bin_cfg)
        + enc_size(&proof.eval_proof, bin_cfg)
        + enc_size(&proof.chiplet_commitments, bin_cfg)
        + enc_size(&proof.chiplet_zerocheck_proofs, bin_cfg)
        + enc_size(&proof.chiplet_logup_aux, bin_cfg)
        + enc_size(&proof.chiplet_eval_proofs, bin_cfg)
        + pad_root_sz
        + outer_sz;

    println!("--------------------------------------------------");
    println!(
        "  Bincode total:          {:>8.2} KB",
        total_sz as f64 / 1024.0
    );
    println!(
        "  Unattributed:           {:>8} bytes",
        total_sz as i64 - itemized as i64
    );
}

/// Prints the zk-Ligero segment, returns
/// the bincode size of `InnerProof::outer`.
fn outer_breakdown<F>(proof: &InnerProof<F>, cfg: bincode::config::Configuration) -> usize
where
    F: TowerField + serde::Serialize,
{
    let segment_sz = enc_size(&proof.outer, cfg);

    let Some(outer) = proof.outer.as_ref() else {
        return segment_sz;
    };

    let aux_root_sz = enc_size(&outer.aux_root, cfg);
    let int_sz = enc_size(&outer.interleaved, cfg);
    let lin_sz = enc_size(&outer.linear, cfg);
    let quad_sz = enc_size(&outer.quadratic, cfg);

    let columns_sz =
        enc_size(&outer.pad_opening.columns, cfg) + enc_size(&outer.aux_opening.columns, cfg);

    let pad_values_sz = enc_size(&outer.pad_opening.values, cfg);
    let pad_path_sz = enc_size(&outer.pad_opening.siblings, cfg);
    let aux_values_sz = enc_size(&outer.aux_opening.values, cfg);
    let aux_path_sz = enc_size(&outer.aux_opening.siblings, cfg);

    println!("  Outer Argument (zk-Ligero):");
    println!("    AUX Root:             {:>8} bytes", aux_root_sz);
    println!("    Interleaved Response: {:>8} bytes", int_sz);
    println!("    Linear Response:      {:>8} bytes", lin_sz);
    println!("    Quadratic Response:   {:>8} bytes", quad_sz);
    println!("    Query Columns:        {:>8} bytes", columns_sz);
    println!("    PAD Opened Values:    {:>8} bytes", pad_values_sz);
    println!("    PAD Merkle Path:      {:>8} bytes", pad_path_sz);
    println!("    AUX Opened Values:    {:>8} bytes", aux_values_sz);
    println!("    AUX Merkle Path:      {:>8} bytes", aux_path_sz);
    println!("    Segment total:        {:>8} bytes", segment_sz);

    segment_sz
}

pub fn result(is_valid: bool) {
    println!("==================================================");

    if is_valid {
        println!("SUCCESS");
    } else {
        println!("FAILURE");
    }
}

/// `HEKATE_ZK=0` proves in the clear,
/// anything else in zero knowledge.
pub fn zero_knowledge() -> bool {
    std::env::var("HEKATE_ZK").as_deref() != Ok("0")
}

#[allow(dead_code)]
pub fn num_vars(default: usize) -> usize {
    std::env::var("HEKATE_NUM_VARS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[allow(dead_code)]
pub fn level(default: &str) -> String {
    std::env::var("HEKATE_LEVEL").unwrap_or_else(|_| default.to_string())
}

/// `HEKATE_PROGRAM_ID=<64 hex>` is the audited constant
/// a deployment pins. Unset, the run derives its own
/// and the verifier's drift check cannot fire.
pub fn audited_id<F: TowerField, P: Program<F>>(program: &P) -> [u8; 32] {
    let Ok(pinned) = std::env::var("HEKATE_PROGRAM_ID") else {
        println!(
            "program_id: {}",
            program_id_hex(program).expect("program_id_hex")
        );

        return program_id(program).expect("program_id");
    };

    decode_id(&pinned).expect("HEKATE_PROGRAM_ID must be 64 lowercase hex chars")
}

fn decode_id(s: &str) -> Option<[u8; 32]> {
    let hex = s.trim().as_bytes();
    if hex.len() != 64 {
        return None;
    }

    let mut out = [0u8; 32];
    for (i, pair) in hex.as_chunks::<2>().0.iter().enumerate() {
        let text = core::str::from_utf8(pair).ok()?;
        out[i] = u8::from_str_radix(text, 16).ok()?;
    }

    Some(out)
}

fn enc_size<T: serde::Serialize>(val: &T, cfg: bincode::config::Configuration) -> usize {
    bincode::serde::encode_to_vec(val, cfg).unwrap().len()
}
