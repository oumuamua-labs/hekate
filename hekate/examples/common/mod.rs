// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate project.
// Copyright (C) 2026 Andrei Kochergin <andrei@oumuamua.dev>
// Copyright (C) 2026 Oumuamua Labs <info@oumuamua.dev>. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use hekate_core::proofs::InnerProof;
use hekate_math::TowerField;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

pub fn init(name: &str) {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug")),
        )
        .try_init();

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

    println!(
        "   Total Proof size: {:.2} KB ({} bytes)\n",
        total_sz as f64 / 1024.0,
        total_sz
    );

    // Main components
    let trace_comm_sz = enc_size(&proof.trace_commitment, bin_cfg);
    let zcheck_sz = enc_size(&proof.zerocheck_proof, bin_cfg);

    // Eval batch argument breakdown
    let eval_sc_sz = enc_size(&proof.eval_proof.sumcheck_proof, bin_cfg);
    let eval_tensor_sz = enc_size(&proof.eval_proof.tensor_vec, bin_cfg);
    let eval_pt_sz = enc_size(&proof.eval_proof.point_evaluations, bin_cfg);
    let ldt_merkle_sz = enc_size(&proof.eval_proof.ldt_proof.ldt_proofs, bin_cfg);
    let ldt_opened_sz = enc_size(&proof.eval_proof.ldt_proof.opened_columns, bin_cfg);

    println!("--------------------------------------------------");
    println!("  PROOF COMPONENT BREAKDOWN");
    println!("--------------------------------------------------");
    println!("  Trace Commitment:       {:>8} bytes", trace_comm_sz);
    println!("  Main AIR ZeroCheck:     {:>8} bytes", zcheck_sz);
    println!("  Eval Batch Argument:");
    println!("    Eval Sumcheck:        {:>8} bytes", eval_sc_sz);
    println!("    Tensor Vector (q):    {:>8} bytes", eval_tensor_sz);
    println!("    Point Evaluations:    {:>8} bytes", eval_pt_sz);
    println!("    LDT Merkle Paths:     {:>8} bytes", ldt_merkle_sz);
    println!("    LDT Opened Columns:   {:>8} bytes", ldt_opened_sz);

    if !proof.chiplet_commitments.is_empty() {
        let n = proof.chiplet_commitments.len();
        let chip_comm_sz = enc_size(&proof.chiplet_commitments, bin_cfg);
        let chip_zc_sz = enc_size(&proof.chiplet_zerocheck_proofs, bin_cfg);
        let chip_eval_sz = enc_size(&proof.chiplet_eval_proofs, bin_cfg);

        println!("  Chiplets ({}):", n);
        println!("    Commitments:          {:>8} bytes", chip_comm_sz);
        println!("    ZeroChecks:           {:>8} bytes", chip_zc_sz);
        println!("    Eval Arguments:       {:>8} bytes", chip_eval_sz);
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

    println!("--------------------------------------------------");
    println!(
        "  TOTAL:                  {:>8.2} KB",
        total_sz as f64 / 1024.0
    );
}

pub fn result(is_valid: bool) {
    println!("==================================================");

    if is_valid {
        println!("SUCCESS");
    } else {
        println!("FAILURE");
    }
}

fn enc_size<T: serde::Serialize>(val: &T, cfg: bincode::config::Configuration) -> usize {
    bincode::serde::encode_to_vec(val, cfg).unwrap().len()
}
