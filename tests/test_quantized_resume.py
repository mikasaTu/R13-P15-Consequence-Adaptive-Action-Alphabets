import numpy as np
import pytest

from caaa_libero.quantization import _validate_resumed_shard_against_plan


FIELDS = {
    "task_id": np.asarray("fixture"),
    "episode_id": np.asarray(12),
    "split": np.asarray("test"),
    "phase": np.asarray("free_space"),
    "snapshot_index": np.asarray(3),
    "methods": np.asarray(["caaa_v2"]),
    "k": np.asarray([64]),
    "direction": np.asarray([0]),
    "sign": np.asarray([1]),
    "radius": np.asarray([0.05]),
    "code_index": np.asarray([2]),
    "decoded_actions": np.zeros((1, 4, 7)),
    "original_actions": np.zeros((1, 4, 7)),
    "original_immediate": np.zeros((1, 46)),
    "original_settled": np.zeros((1, 46)),
    "original_mask": np.ones((46,), dtype=bool),
    "original_contact_mode": np.asarray([0]),
    "original_settled_progress": np.asarray([0.25]),
    "original_settled_success": np.asarray([0]),
}


def test_quantized_resume_requires_exact_current_plan(tmp_path):
    plan = tmp_path / "plan.npz"
    shard = tmp_path / "shard.npz"
    np.savez(plan, **FIELDS)
    np.savez(shard, **FIELDS, settled=np.zeros((1, 46)))
    _validate_resumed_shard_against_plan(str(plan), str(shard))

    drifted = dict(FIELDS)
    drifted["decoded_actions"] = np.ones((1, 4, 7))
    np.savez(plan, **drifted)
    with pytest.raises(RuntimeError, match="does not match current frozen plan"):
        _validate_resumed_shard_against_plan(str(plan), str(shard))
