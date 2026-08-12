#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

project_root=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r13-p15-caaa-v2-libero
stage1_launcher=${project_root}/scripts/run_stage1_pai.sh
expected_project_commit=34995e8e7c3069b22785ad04536f0d429e75c0fc
expected_project_tree=ad6fa59b782f63624ee3ccef8e880a2398669ce8
expected_stage1_launcher_sha256=569d031517a581a080c3a68a46a67df76432318d5c9e35edb1983e7f3c617e6a

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
