default:
    @just --list --unsorted

publish-all:
    #!/usr/bin/env bash
    set -euo pipefail
    crates=(
        hekate-crypto
        hekate-core
        hekate-program
        hekate-gadgets
        hekate-verifier
        hekate-sdk
        hekate-prover-sys
        hekate-scribble
        hekate-keccak
        hekate-aes
        hekate-pqc
    )
    for c in "${crates[@]}"; do
        case "$c" in
            hekate-prover-sys)
                cargo publish -p "$c" --features ct ;;
            *)
                cargo publish -p "$c" ;;
        esac
        cargo update
    done
