#!/usr/bin/env bash
# Regenerate R-INLA reference fixtures.
#
# Usage:
#   bench/Rrun.sh                # all examples
#   bench/Rrun.sh 04_salamander_mating
#
# Each examples/<id>/rinla.R is expected to write a JSON file to
# test/fixtures/<id>/rinla_reference.json. The shape of that JSON is what
# test/parity/parity_helpers.jl reads — see that file for the contract.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

if [[ $# -gt 0 ]]; then
    targets=("$@")
else
    targets=()
    for d in "$ROOT"/examples/*/; do
        if [[ -f "$d/rinla.R" ]]; then
            targets+=("$(basename "$d")")
        fi
    done
fi

if [[ ${#targets[@]} -eq 0 ]]; then
    echo "no examples with rinla.R found" >&2
    exit 1
fi

for id in "${targets[@]}"; do
    script="$ROOT/examples/$id/rinla.R"
    if [[ ! -f "$script" ]]; then
        echo "skip $id (no rinla.R)"
        continue
    fi
    mkdir -p "$ROOT/test/fixtures/$id"
    echo "==> $id"
    Rscript "$script"
done
