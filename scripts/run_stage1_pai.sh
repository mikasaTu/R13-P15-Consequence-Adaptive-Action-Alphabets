#!/usr/bin/env bash
set -euo pipefail

umask 077

project_root=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r13-p15-caaa-v2-libero
output_root=/mnt/cpfs/zbl-cpfs-new/dataset/leon/experiments/r13_p15_caaa_v2/stage1
libero_source=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/LIBERO-original
simulation_python=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero-original/bin/python
analysis_python=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/openpi_py311/bin/python
artifact_dir=${PAI_CANARY_RUN_DIR:?PAI_CANARY_RUN_DIR is required}
run_id=${PAI_CANARY_RUN_ID:?PAI_CANARY_RUN_ID is required}
nonce=${PAI_CANARY_NONCE:?PAI_CANARY_NONCE is required}
expected_project_commit=${CAAA_EXPECTED_PROJECT_COMMIT:?CAAA_EXPECTED_PROJECT_COMMIT is required}
expected_project_tree=${CAAA_EXPECTED_PROJECT_TREE:?CAAA_EXPECTED_PROJECT_TREE is required}

on_error() {
  local exit_code=$?
  printf 'CAAA_STAGE1_COMMAND_FAILED line=%s exit_code=%s command=%q\n' \
    "${BASH_LINENO[0]:-unknown}" "${exit_code}" "${BASH_COMMAND}" >&2
  return "${exit_code}"
}
trap on_error ERR

for required in git nvidia-smi realpath sha256sum stat sync; do
  command -v "${required}" >/dev/null
done
test "$(id -u):$(id -g)" = 2254:2254
test "${PAI_CANARY_EXPECTED_GPUS:-}" = 8
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA A800')" = 8
[[ "${run_id}" =~ ^[a-z0-9][a-z0-9-]{2,63}$ ]]
[[ "${nonce}" =~ ^[a-f0-9]{32}$ ]]
case "${artifact_dir}" in
  /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r13-p15-caaa-v2-stage1-pai/*) ;;
  *) printf 'artifact directory escaped the authorized run root\n' >&2; exit 71 ;;
esac
test "$(realpath -e "${artifact_dir}")" = "${artifact_dir}"
test "$(stat -c '%u:%g' "${artifact_dir}")" = 2254:2254
test "$(realpath -e "${output_root}")" = "${output_root}"
test "$(stat -c '%u:%g' "${output_root}")" = 2254:2254
test "$(git -C "${project_root}" rev-parse HEAD)" = "${expected_project_commit}"
test "$(git -C "${project_root}" rev-parse 'HEAD^{tree}')" = "${expected_project_tree}"
test -z "$(git -C "${project_root}" status --porcelain)"

export PYTHONPATH="${project_root}:${libero_source}"
export MUJOCO_GL=glx
export CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME="${artifact_dir}/cache/xdg"
export PYTHONPYCACHEPREFIX="${artifact_dir}/cache/pycache"
export TMPDIR="${artifact_dir}/tmp"
mkdir -p "${XDG_CACHE_HOME}" "${PYTHONPYCACHEPREFIX}" "${TMPDIR}"
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
  if [[ "${phase}" = branches ]]; then
    first_marker=
    marker_waits=0
    while [[ -z "${first_marker}" ]]; do
      first_marker=$(find "${output_root}/work/branch_shards" -type f -name '*.complete.json' -print -quit 2>/dev/null || true)
      if [[ -z "${first_marker}" ]]; then
        any_alive=0
        for pid in "${pids[@]}"; do
          if kill -0 "${pid}" 2>/dev/null; then
            any_alive=1
          fi
        done
        if [[ "${any_alive}" -eq 0 ]]; then
          printf 'all branch workers exited before a committed shard appeared\n' >&2
          return 1
        fi
        marker_waits=$((marker_waits + 1))
        if [[ "${marker_waits}" -ge 360 ]]; then
          printf 'no committed branch shard appeared within one hour\n' >&2
          return 1
        fi
        sleep 10
      fi
    done
    export PAI_FIRST_MARKER="${first_marker}"
    export PAI_STAGE1_ARTIFACT_DIR="${artifact_dir}"
    export PAI_STAGE1_RUN_ID="${run_id}"
    export PAI_STAGE1_NONCE="${nonce}"
    "${simulation_python}" - <<'PY'
import json
import os

from caaa_libero.pipeline import utc_now
from caaa_libero.storage import atomic_json, validate_complete

marker = os.environ["PAI_FIRST_MARKER"]
payload = marker[: -len(".complete.json")]
valid, metadata = validate_complete(payload)
if not valid:
    raise RuntimeError("first shard failed hash validation: %s" % (metadata,))
atomic_json(
    os.path.join(os.environ["PAI_STAGE1_ARTIFACT_DIR"], "FIRST_COMMITTED_SHARD.json"),
    {
        "created_utc": utc_now(),
        "run_id": os.environ["PAI_STAGE1_RUN_ID"],
        "nonce": os.environ["PAI_STAGE1_NONCE"],
        "payload": payload,
        "completion_marker": marker,
        "payload_sha256": metadata["payload_sha256"],
        "status": "persisted_committed_sample_or_shard",
    },
)
PY
    sync -f "${artifact_dir}/FIRST_COMMITTED_SHARD.json"
  fi
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

export PAI_STAGE1_ARTIFACT_DIR="${artifact_dir}"
export PAI_STAGE1_RUN_ID="${run_id}"
export PAI_STAGE1_NONCE="${nonce}"
export PAI_STAGE1_OUTPUT_ROOT="${output_root}"
"${analysis_python}" - <<'PY'
import json
import os

from caaa_libero.pipeline import utc_now
from caaa_libero.storage import atomic_json, sha256_file

root = os.environ["PAI_STAGE1_OUTPUT_ROOT"]
with open(os.path.join(root, "work", "finalize_manifest.json"), "r", encoding="utf-8") as handle:
    finalized = json.load(handle)
if finalized.get("status") != "STAGE1_COMPLETE":
    raise RuntimeError("finalize manifest is not complete")
atomic_json(
    os.path.join(os.environ["PAI_STAGE1_ARTIFACT_DIR"], "STAGE1_COMPLETE.json"),
    {
        "created_utc": utc_now(),
        "run_id": os.environ["PAI_STAGE1_RUN_ID"],
        "nonce": os.environ["PAI_STAGE1_NONCE"],
        "formal_output_root": root,
        "report_sha256": sha256_file(os.path.join(root, "STAGE1_REPORT.md")),
        "disposition": finalized["disposition"],
        "quantized_rows": finalized["quantized_rows"],
        "status": "STAGE1_COMPLETE",
    },
)
PY
sync -f "${artifact_dir}/STAGE1_COMPLETE.json"

echo "STAGE1_PIPELINE_COMPLETE ${output_root}"
