default:
    @just --list --unsorted

publish-all:
    #!/usr/bin/env bash
    set -euo pipefail
    crates=(
        hekate-crypto
        hekate-core
        hekate-program
        hekate-verifier
        hekate-sdk
        hekate-prover-sys
        hekate-scribble
        hekate-gadgets
        hekate-keccak
        hekate-aes
        hekate-sha2
        hekate-rsa
        hekate-pqc
    )

    index_path() {
        local n="$1"
        case ${#n} in
            1) printf '1/%s' "$n" ;;
            2) printf '2/%s' "$n" ;;
            3) printf '3/%s/%s' "${n:0:1}" "$n" ;;
            *) printf '%s/%s/%s' "${n:0:2}" "${n:2:2}" "$n" ;;
        esac
    }

    already_published() {
        curl -sf --max-time 20 "https://index.crates.io/$(index_path "$1")" \
            | grep -q "\"vers\":\"$2\""
    }

    versions="$(cargo metadata --no-deps --format-version 1)"

    for c in "${crates[@]}"; do
        v="$(printf '%s' "$versions" | python3 -c \
            "import sys,json;print(next(p['version'] for p in json.load(sys.stdin)['packages'] if p['name']=='$c'))")"

        if already_published "$c" "$v"; then
            echo "skip $c $v, already on crates.io" >&2
            continue
        fi

        case "$c" in
            hekate-prover-sys)
                cargo publish -p "$c" --features ct ;;
            *)
                cargo publish -p "$c" ;;
        esac
        cargo update
    done

example name variant="ct":
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

    /usr/bin/time -l target/release/examples/{{name}} &
    tpid=$!

    pid=""
    while [ -z "$pid" ] && kill -0 "$tpid" 2>/dev/null; do
        pid=$(pgrep -P "$tpid" 2>/dev/null | head -1) || true
    done
    pid="${pid:-$tpid}"

    # phys_footprint_peak counts compressed pages, ru_maxrss does not.
    peak=""
    while kill -0 "$pid" 2>/dev/null; do
        s=$(footprint -p "$pid" 2>/dev/null | awk '/phys_footprint_peak:/{print $2, $3}') || true
        if [ -n "$s" ]; then peak="$s"; fi
        sleep 0.05
    done

    wait "$tpid"
    echo "Peak memory: ${peak:-unavailable}"
