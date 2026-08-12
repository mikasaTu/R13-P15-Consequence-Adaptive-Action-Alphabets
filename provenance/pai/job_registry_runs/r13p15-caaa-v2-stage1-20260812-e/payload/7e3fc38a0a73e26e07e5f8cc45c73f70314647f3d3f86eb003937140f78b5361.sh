#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

project_root=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r13-p15-caaa-v2-libero
stage1_launcher=${project_root}/scripts/run_stage1_pai.sh
expected_project_commit=b9f9b690e7fc59c33a9e95eb4974ac654115b9c3
expected_project_tree=28268645f44a8de44fef4d1a182fffe1e5c2a143
expected_stage1_launcher_sha256=6b7f52c8e6cb614d810f96577289cf06369a2d0b627e9a6ba894b8409c00167c

for required in git sha256sum; do
  command -v "${required}" >/dev/null
done
test "$(id -u):$(id -g)" = 2254:2254
test "$(git -C "${project_root}" rev-parse HEAD)" = "${expected_project_commit}"
test "$(git -C "${project_root}" rev-parse 'HEAD^{tree}')" = "${expected_project_tree}"
test -z "$(git -C "${project_root}" status --porcelain)"
test "$(sha256sum "${stage1_launcher}" | awk '{print $1}')" = "${expected_stage1_launcher_sha256}"

export CAAA_EXPECTED_PROJECT_COMMIT="${expected_project_commit}"
export CAAA_EXPECTED_PROJECT_TREE="${expected_project_tree}"
exec "${stage1_launcher}"
