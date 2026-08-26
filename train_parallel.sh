#!/usr/bin/env bash

set -euo pipefail

master_addr=localhost
master_port=24105
forwarded_args=()

for arg in "$@"; do
  case "$arg" in
    --port=*) master_port="${arg#*=}" ;;
    *) forwarded_args+=("$arg") ;;
  esac
done

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must list the GPUs to use." >&2
  exit 2
fi

IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
num_gpus=${#gpu_ids[@]}
cpu_count=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)
omp_threads=$((cpu_count / num_gpus))
if ((omp_threads < 1)); then
  omp_threads=1
fi
if ((omp_threads > 64)); then
  omp_threads=64
fi

OMP_NUM_THREADS=$omp_threads torchrun \
  --nproc_per_node "$num_gpus" \
  --rdzv_endpoint "$master_addr:$master_port" \
  train.py "${forwarded_args[@]}"
