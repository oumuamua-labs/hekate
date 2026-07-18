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

example name arg="" variant="ct":
    #!/usr/bin/env bash
    set -euo pipefail

    case "{{variant}}" in
        ct)     feats="std parallel blake3 ct" ;;
        public) feats="std parallel blake3 table-math public" ;;
        *)      echo "variant must be ct or public, got '{{variant}}'" >&2; exit 2 ;;
    esac

    cargo build --release -p hekate \
        --no-default-features --features "$feats" \
        --example {{name}}

    target/release/examples/{{name}} {{arg}} &
    pid=$!

    # phys_footprint_peak counts compressed pages, ru_maxrss does not.
    peak=""
    while kill -0 "$pid" 2>/dev/null; do
        s=$(footprint -p "$pid" 2>/dev/null | awk '/phys_footprint_peak:/{print $2, $3}') || true
        if [ -n "$s" ]; then peak="$s"; fi
        sleep 0.5
    done

    wait "$pid"
    echo "Peak memory: ${peak:-unavailable}"
