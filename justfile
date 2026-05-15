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
        hekate
    )
    for c in "${crates[@]}"; do
        if [ "$c" = "hekate-prover-sys" ]; then
            cargo publish -p "$c" --features ct
        else
            cargo publish -p "$c"
        fi
        cargo update
    done
