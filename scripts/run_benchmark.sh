#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${1:-${repo_dir}/result/runs}"
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${output_root}/${run_id}"
mkdir -p "${run_dir}"
cd "${repo_dir}"

make
{
  echo "utc=${run_id}"
  echo "git_revision=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty=$(test -n "$(git status --porcelain 2>/dev/null)" && echo true || echo false)"
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,power.limit --format=csv,noheader
  "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" --version
} > "${run_dir}/environment.txt"

./build/attention_bench \
  --min-n 1024 --max-n 62464 --step 4096 \
  --kernel V4 --target-ms 200 --check-rows 32 \
  --output "${run_dir}/benchmark.csv" \
  2> "${run_dir}/validation.log"

python3 plot_benchmark.py "${run_dir}/benchmark.csv" "${run_dir}/performance.png"
echo "Benchmark artifacts: ${run_dir}"
