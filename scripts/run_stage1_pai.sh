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
export LIBERO_CONFIG_PATH="${project_root}/config/libero"
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
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    return 1
  fi
  if [[ "${phase}" = branches ]]; then
    export PAI_STAGE1_ARTIFACT_DIR="${artifact_dir}"
    export PAI_STAGE1_RUN_ID="${run_id}"
    export PAI_STAGE1_NONCE="${nonce}"
    export PAI_STAGE1_OUTPUT_ROOT="${output_root}"
    "${simulation_python}" - <<'PY'
import glob
import json
import os

from caaa_libero.pipeline import utc_now
from caaa_libero.storage import atomic_json, validate_complete

root = os.environ["PAI_STAGE1_OUTPUT_ROOT"]
run_id = os.environ["PAI_STAGE1_RUN_ID"]
nonce = os.environ["PAI_STAGE1_NONCE"]
tasks = ("bowl_on_plate", "plate_push", "stove_turn_on", "wine_rack")
manifest_paths = []
for task in tasks:
    path = os.path.join(root, "work", "branch_collection_%s.json" % task)
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("pai_run_id") != run_id or manifest.get("pai_nonce") != nonce:
        raise RuntimeError("branch manifest is not bound to the current launch: %s" % path)
    if manifest.get("count") != 64:
        raise RuntimeError("branch manifest is incomplete: %s" % path)
    manifest_paths.append(path)
shards = sorted(glob.glob(os.path.join(root, "work", "branch_shards", "*", "*.npz")))
if len(shards) != 256:
    raise RuntimeError("expected 256 branch shards, found %d" % len(shards))
payload = shards[0]
valid, metadata = validate_complete(payload)
if not valid:
    raise RuntimeError("committed shard failed hash validation: %s" % (metadata,))
atomic_json(
    os.path.join(os.environ["PAI_STAGE1_ARTIFACT_DIR"], "FIRST_COMMITTED_SHARD.json"),
    {
        "created_utc": utc_now(),
        "run_id": run_id,
        "nonce": nonce,
        "payload": payload,
        "completion_marker": payload + ".complete.json",
        "payload_sha256": metadata["payload_sha256"],
        "current_launch_manifests": manifest_paths,
        "current_launch_manifest_status": "updated_after_hash_verified_collection_or_resume",
        "status": "persisted_committed_sample_or_shard",
    },
)
PY
    sync -f "${artifact_dir}/FIRST_COMMITTED_SHARD.json"
  elif [[ "${phase}" = quantized ]]; then
    export PAI_STAGE1_RUN_ID="${run_id}"
    export PAI_STAGE1_NONCE="${nonce}"
    export PAI_STAGE1_OUTPUT_ROOT="${output_root}"
    "${simulation_python}" - <<'PY'
import glob
import json
import os

from caaa_libero.storage import validate_complete

root = os.environ["PAI_STAGE1_OUTPUT_ROOT"]
run_id = os.environ["PAI_STAGE1_RUN_ID"]
nonce = os.environ["PAI_STAGE1_NONCE"]
tasks = ("bowl_on_plate", "plate_push", "stove_turn_on", "wine_rack")
for task in tasks:
    path = os.path.join(root, "work", "quantized_collection_%s.json" % task)
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("pai_run_id") != run_id or manifest.get("pai_nonce") != nonce:
        raise RuntimeError("quantized manifest is not bound to current launch: %s" % path)
    if manifest.get("count") != 32:
        raise RuntimeError("quantized manifest is incomplete: %s" % path)
shards = sorted(glob.glob(os.path.join(root, "work", "quantized_shards", "*", "*.npz")))
if len(shards) != 128:
    raise RuntimeError("expected 128 quantized shards, found %d" % len(shards))
for payload in shards:
    valid, metadata = validate_complete(payload)
    if not valid:
        raise RuntimeError("committed quantized shard failed hash validation: %s %s" % (payload, metadata))
PY
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
