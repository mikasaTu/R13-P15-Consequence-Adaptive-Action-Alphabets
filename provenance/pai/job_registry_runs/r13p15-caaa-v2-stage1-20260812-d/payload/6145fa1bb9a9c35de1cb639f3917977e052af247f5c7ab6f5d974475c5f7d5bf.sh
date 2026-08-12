#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

project_root=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r13-p15-caaa-v2-libero
stage1_launcher=${project_root}/scripts/run_stage1_pai.sh
expected_project_commit=ef5aa46daa43fadf05f9cb1bbb919d4bb6cfe2fb
expected_project_tree=55fd7e237606140d5968d2f77cfe7655e2ae4668
expected_stage1_launcher_sha256=a8f2367241b3f82cec83a90b743748169c7fa5c6d72d074ebc6842fe846d6628

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
