#!/usr/bin/env bash
set -euo pipefail

project_root=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r13-p15-caaa-v2-libero
output_root=/mnt/cpfs/zbl-cpfs-new/dataset/leon/experiments/r13_p15_caaa_v2/stage1
libero_source=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/LIBERO-original
simulation_python=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero-original/bin/python
analysis_python=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/openpi_py311/bin/python

export PYTHONPATH="${project_root}:${libero_source}"
export MUJOCO_GL=glx
export CUDA_VISIBLE_DEVICES=0
mkdir -p "${output_root}/work/logs"

common=(--output-root "${output_root}" --libero-source "${libero_source}")

"${simulation_python}" -m caaa_libero.cli formal-setup "${common[@]}" \
  >"${output_root}/work/logs/formal_setup.log" 2>&1

run_task_phase() {
  local phase=$1
  local python_executable=$2
  local command=$3
  local failed=0
  local pids=()
  local task
  for task in bowl_on_plate plate_push stove_turn_on wine_rack; do
    "${python_executable}" -m caaa_libero.cli "${command}" "${common[@]}" --task-id "${task}" \
      >"${output_root}/work/logs/${phase}_${task}.log" 2>&1 &
    pids+=("$!")
  done
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    return 1
  fi
}

run_task_phase branches "${simulation_python}" collect-branches

"${analysis_python}" -m caaa_libero.cli prepare-analysis "${common[@]}" \
  >"${output_root}/work/logs/prepare_analysis.log" 2>&1

run_task_phase quantized "${simulation_python}" collect-quantized

"${analysis_python}" -m caaa_libero.cli finalize "${common[@]}" \
  >"${output_root}/work/logs/finalize.log" 2>&1

echo "STAGE1_PIPELINE_COMPLETE ${output_root}"
