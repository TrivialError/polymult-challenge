#!/usr/bin/env bash
# Build solve.cu as plain C++ against pm_emu.h (no GPU needed) and check it
# against tests/oracle.py on a set of points.
#
#   tests/run_emu.sh                 quick set
#   tests/run_emu.sh full            also the two challenge points
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p build-emu
${CXX:-clang++} -std=c++17 -O2 -Wall -Wextra -pthread -I. \
    -x c++ solve.cu -x none main.cpp -o build-emu/solution_emu

python3 tests/oracle.py selftest

points=(
  "64 32 4 mixed"
  "16 33 4 mixed"
  "2 40 2 mixed"
  "1024 512 4 mixed"
  "256 1747 4 mixed"
  "4096 100 2 mixed"
  "8192 256 4 mixed"
  "2048 868 2 mixed"
  "512 4000 2 mixed"
)
if [[ "${1:-}" == "full" ]]; then
  points+=("32768 868 2 mixed" "65536 1747 2 mixed")
fi

fail=0
for spec in "${points[@]}"; do
  read -r N W C M <<<"$spec"
  dir="build-emu/pt_${N}_${W}"
  if [[ ! -f "$dir/manifest.json" ]]; then
    rm -rf "$dir"
    python3 tests/oracle.py make "$dir" "$N" "$W" "$C" "$M"
  fi
  rm -rf "$dir/out"
  ./build-emu/solution_emu "$dir"
  python3 tests/oracle.py verify "$dir" || fail=1
done
exit $fail
