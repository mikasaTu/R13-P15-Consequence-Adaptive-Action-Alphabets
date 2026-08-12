from pathlib import Path

import yaml


def test_pai_launcher_uses_noninteractive_libero_config():
    project_root = Path(__file__).resolve().parents[1]
    launcher = (project_root / "scripts" / "run_stage1_pai.sh").read_text()
    assert 'export LIBERO_CONFIG_PATH="${project_root}/config/libero"' in launcher

    config = yaml.safe_load(
        (project_root / "config" / "libero" / "config.yaml").read_text()
    )
    assert set(config) == {
        "assets",
        "bddl_files",
        "benchmark_root",
        "datasets",
        "init_states",
    }
    for path in config.values():
        assert path.startswith("/mnt/cpfs/zbl-cpfs-new/")
