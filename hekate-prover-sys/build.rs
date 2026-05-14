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

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use ed25519_dalek::pkcs8::DecodePublicKey as _;
use ed25519_dalek::{Signature as EdSignature, VerifyingKey as EdKey};
use pqcrypto_mldsa::mldsa65;
use pqcrypto_traits::sign::{
    DetachedSignature as _, PublicKey as _, VerificationError as MlVerifyErr,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use spki::{SubjectPublicKeyInfoOwned, der::Decode};
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;
use ureq::Agent;

#[derive(Debug, Deserialize)]
struct Manifest {
    schema_version: u32,
    hekate_version: String,
    publisher: Publisher,
    targets: Vec<Target>,
}

#[derive(Debug, Deserialize)]
struct Publisher {
    ed25519_pubkey_pem: String,
    mldsa65_pubkey_pem: String,
}

#[derive(Debug, Deserialize)]
struct Target {
    variant: String,
    triple: String,
    filename: String,
    sha256: String,
    ed25519_sig: String,
    mldsa65_sig: String,
    url: String,
}

fn pick_variant() -> Result<&'static str, Box<dyn std::error::Error>> {
    let ct = env::var_os("CARGO_FEATURE_CT").is_some();
    let public = env::var_os("CARGO_FEATURE_PUBLIC").is_some();

    match (ct, public) {
        (true, false) => Ok("ct"),
        (false, true) => Ok("public"),
        (false, false) => Err("hekate-prover-sys: enable exactly one of features `ct` or `public`. \
                               Add `features = [\"ct\"]` (constant-time, for private witnesses) or \
                               `features = [\"public\"]` (variable-time table-math, public data only) \
                               to your Cargo.toml."
            .into()),
        (true, true) => Err("hekate-prover-sys: features `ct` and `public` are mutually exclusive. \
                             Enable exactly one."
            .into()),
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    if env::var_os("DOCS_RS").is_some() {
        return Ok(());
    }

    let manifest_path = Path::new(&env::var("CARGO_MANIFEST_DIR")?).join("artifacts/manifest.toml");
    let manifest_str = fs::read_to_string(&manifest_path)
        .map_err(|e| format!("read manifest at {}: {e}", manifest_path.display()))?;
    let manifest: Manifest = toml::from_str(&manifest_str)?;

    if manifest.schema_version != 2 {
        return Err(format!(
            "manifest schema_version = {}, expected 2",
            manifest.schema_version
        )
        .into());
    }

    let variant = pick_variant()?;
    let triple = env::var("TARGET")?;
    let target = manifest
        .targets
        .iter()
        .find(|t| t.triple == triple && t.variant == variant)
        .ok_or_else(|| {
            format!(
                "no manifest entry for variant {variant} / triple {triple} (manifest version {})",
                manifest.hekate_version
            )
        })?;

    let cdylib = locate_cdylib(&manifest, target)?;
    let bytes =
        fs::read(&cdylib).map_err(|e| format!("read cdylib at {}: {e}", cdylib.display()))?;

    verify_sha256(&bytes, &target.sha256)?;
    verify_ed25519(
        &bytes,
        &target.ed25519_sig,
        &manifest.publisher.ed25519_pubkey_pem,
    )?;
    verify_mldsa65(
        &bytes,
        &target.mldsa65_sig,
        &manifest.publisher.mldsa65_pubkey_pem,
    )?;

    let staged_dir = stage_into_outdir(&cdylib, &target.filename)?;

    println!("cargo:rustc-link-search=native={}", staged_dir.display());
    println!("cargo:rustc-link-lib=dylib=hekate_prover_cdylib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", staged_dir.display());

    Ok(())
}

fn locate_cdylib(
    manifest: &Manifest,
    target: &Target,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Ok(dir) = env::var("HEKATE_PROVER_DYLIB_DIR") {
        let p = Path::new(&dir).join(&target.filename);
        if p.is_file() {
            return Ok(p);
        }

        return Err(format!(
            "HEKATE_PROVER_DYLIB_DIR set to {dir} but {} does not exist there",
            target.filename
        )
        .into());
    }

    let cache = cache_path(manifest, target)?;
    if cache.is_file() {
        let bytes = fs::read(&cache)?;
        let actual = Sha256::digest(&bytes);
        let expected = hex::decode(target.sha256.trim()).map_err(|e| format!("sha256 hex: {e}"))?;

        if expected.as_slice() == actual.as_slice() {
            return Ok(cache);
        }

        fs::remove_file(&cache)?;
    }

    download_into(&cache, &target.url)?;

    Ok(cache)
}

fn cache_path(manifest: &Manifest, target: &Target) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let root = match env::var("HEKATE_PROVER_CACHE_DIR") {
        Ok(v) => PathBuf::from(v),
        Err(_) => {
            let home = env::var("HOME").or_else(|_| env::var("USERPROFILE"))?;
            PathBuf::from(home).join(".cache").join("hekate-prover-sys")
        }
    };

    let dir = root
        .join(&manifest.hekate_version)
        .join(&target.variant)
        .join(&target.triple);

    fs::create_dir_all(&dir)?;

    Ok(dir.join(&target.filename))
}

fn download_into(dest: &Path, url: &str) -> Result<(), Box<dyn std::error::Error>> {
    let agent: Agent = Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(15)))
        .timeout_recv_body(Some(Duration::from_secs(120)))
        .timeout_send_body(Some(Duration::from_secs(60)))
        .build()
        .into();

    let mut resp = agent
        .get(url)
        .call()
        .map_err(|e| format!("download {url}: {e}"))?;

    if resp.status().as_u16() != 200 {
        return Err(format!("download {url}: HTTP {}", resp.status()).into());
    }

    let mut reader = resp.body_mut().as_reader();

    let tmp = dest.with_extension(format!("partial.{}", std::process::id()));

    let mut f = fs::File::create(&tmp)?;
    let mut buf = [0u8; 64 * 1024];

    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }

        f.write_all(&buf[..n])?;
    }

    f.sync_all()?;
    drop(f);
    fs::rename(&tmp, dest)?;

    Ok(())
}

fn verify_sha256(bytes: &[u8], expected_hex: &str) -> Result<(), Box<dyn std::error::Error>> {
    let actual = Sha256::digest(bytes);
    let expected = hex::decode(expected_hex.trim()).map_err(|e| format!("sha256 hex: {e}"))?;

    if expected.as_slice() != actual.as_slice() {
        return Err(format!(
            "SHA-256 mismatch: expected {expected_hex}, got {}",
            hex::encode(actual)
        )
        .into());
    }

    Ok(())
}

fn verify_ed25519(
    msg: &[u8],
    sig_b64: &str,
    pubkey_pem: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let key = EdKey::from_public_key_pem(pubkey_pem.trim())
        .map_err(|e| format!("Ed25519 pubkey PEM: {e}"))?;

    let sig_bytes = B64
        .decode(sig_b64.trim())
        .map_err(|e| format!("Ed25519 sig base64: {e}"))?;
    let sig =
        EdSignature::from_slice(&sig_bytes).map_err(|e| format!("Ed25519 sig length: {e}"))?;

    key.verify_strict(msg, &sig)
        .map_err(|e| format!("Ed25519 verification failed: {e}"))?;

    Ok(())
}

fn verify_mldsa65(
    msg: &[u8],
    sig_b64: &str,
    pubkey_pem: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let raw_pub = spki_subject_public_key(pubkey_pem)?;
    let key = mldsa65::PublicKey::from_bytes(&raw_pub)
        .map_err(|e| format!("ML-DSA-65 pubkey from raw bytes: {e:?}"))?;

    let sig_bytes = B64
        .decode(sig_b64.trim())
        .map_err(|e| format!("ML-DSA-65 sig base64: {e}"))?;
    let sig = mldsa65::DetachedSignature::from_bytes(&sig_bytes)
        .map_err(|e| format!("ML-DSA-65 sig from bytes: {e:?}"))?;

    mldsa65::verify_detached_signature(&sig, msg, &key).map_err(|e: MlVerifyErr| {
        Box::<dyn std::error::Error>::from(format!("ML-DSA-65 verification failed: {e:?}"))
    })?;

    Ok(())
}

fn spki_subject_public_key(pem: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let body: String = pem
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with("-----"))
        .collect::<Vec<&str>>()
        .join("");
    let der = B64
        .decode(body)
        .map_err(|e| format!("PEM body base64: {e}"))?;
    let spki = SubjectPublicKeyInfoOwned::from_der(&der).map_err(|e| format!("SPKI DER: {e}"))?;

    let bytes = spki
        .subject_public_key
        .as_bytes()
        .ok_or("SPKI subjectPublicKey is not byte-aligned")?;

    Ok(bytes.to_vec())
}

fn stage_into_outdir(src: &Path, filename: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let out = PathBuf::from(env::var("OUT_DIR")?);
    let dest = out.join(filename);
    fs::copy(src, &dest)?;

    Ok(out)
}

fn main() {
    println!("cargo:rerun-if-changed=artifacts/manifest.toml");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=HEKATE_PROVER_DYLIB_DIR");
    println!("cargo:rerun-if-env-changed=HEKATE_PROVER_CACHE_DIR");
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    if let Err(e) = run() {
        eprintln!("hekate-prover-sys build failure: {e:#}");
        std::process::exit(1);
    }
}
