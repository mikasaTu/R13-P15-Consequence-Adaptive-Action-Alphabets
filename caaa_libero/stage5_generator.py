"""Frozen state-only nominal H=4 behavior-cloning generator for Stage 5.

The generator is deliberately separate from the R13-P15 metric.  Collection
reads only official demonstrations 0--31 and physical observations available
at decision time.  The exported NPZ can be evaluated with NumPy in the older
LIBERO simulator environment, avoiding a cross-version torch dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

from . import config
from .env_adapter import FEATURE_NAMES, LiberoTaskRuntime
from .stage5_config import (
    GENERATOR_ARCHITECTURE,
    GENERATOR_TRAIN_EPISODES,
    GENERATOR_TRAIN_SEED,
    OUTPUT_RELATIVE,
    SCRATCH_ROOT,
    TASKS,
)
from .storage import atomic_json, atomic_npz, mark_complete, sha256_file, validate_complete


DATA_SCHEMA = "stage5-state-h4-bc-training-v1"
CHECKPOINT_SCHEMA = "stage5-state-h4-bc-numpy-v1"
INPUT_DIM = len(FEATURE_NAMES) * 2 + config.ACTION_DIM + len(TASKS)
OUTPUT_DIM = config.CHUNK_HORIZON * config.ACTION_DIM


def training_data_path(scratch_root=SCRATCH_ROOT):
    return os.path.join(scratch_root, "nominal_generator", "training_data.npz")


def checkpoint_path(output_root):
    return os.path.join(output_root, "nominal_generator", "state_h4_bc.npz")


def _load_complete_npz(path, schema):
    valid, evidence = validate_complete(path)
    if not valid:
        raise RuntimeError("incomplete artifact %s: %s" % (path, evidence))
    with np.load(path, allow_pickle=False) as data:
        output = {key: np.asarray(data[key]).copy() for key in data.files}
    if str(output["schema_version"].item()) != schema:
        raise RuntimeError("schema mismatch for " + path)
    return output


def collect_training_data(
    libero_source=config.LIBERO_SOURCE_DEFAULT,
    dataset_root=config.DATASET_ROOT_DEFAULT,
    scratch_root=SCRATCH_ROOT,
):
    """Extract every valid H=4 point without reading any Stage 5 label."""
    destination = training_data_path(scratch_root)
    valid, _ = validate_complete(destination)
    if valid:
        return {
            "path": destination,
            "sha256": sha256_file(destination),
            "reused": True,
        }
    state_rows = []
    mask_rows = []
    previous_rows = []
    task_rows = []
    action_rows = []
    episode_rows = []
    timestep_rows = []
    task_name_rows = []
    started = time.time()
    for task_index, task in enumerate(TASKS):
        runtime = LiberoTaskRuntime(task, libero_source, dataset_root)
        try:
            one_hot = np.eye(len(TASKS), dtype=np.float32)[task_index]
            for episode_id in GENERATOR_TRAIN_EPISODES:
                episode = runtime.load_episode(episode_id)
                runtime.initialize_episode_model(episode)
                actions = np.asarray(episode["actions"], dtype=np.float64)
                states = np.asarray(episode["states"], dtype=np.float64)
                count = min(len(states), len(actions) - config.CHUNK_HORIZON + 1)
                for timestep in range(max(count, 0)):
                    runtime.env.sim.set_state_from_flattened(states[timestep])
                    runtime.env.sim.forward()
                    runtime.env._post_process()
                    measured = runtime.measure()
                    state_rows.append(np.asarray(measured["vector"], dtype=np.float32))
                    mask_rows.append(np.asarray(measured["mask"], dtype=bool))
                    previous_rows.append(
                        np.zeros(config.ACTION_DIM, dtype=np.float32)
                        if timestep == 0
                        else np.asarray(actions[timestep - 1], dtype=np.float32)
                    )
                    task_rows.append(one_hot)
                    action_rows.append(
                        np.asarray(
                            actions[timestep : timestep + config.CHUNK_HORIZON],
                            dtype=np.float32,
                        ).reshape(-1)
                    )
                    episode_rows.append(int(episode_id))
                    timestep_rows.append(int(timestep))
                    task_name_rows.append(str(task["task_id"]))
                print(
                    "generator-data task=%s episode=%d rows=%d elapsed=%.1fs"
                    % (task["task_id"], episode_id, len(state_rows), time.time() - started),
                    flush=True,
                )
        finally:
            runtime.close()
    state = np.asarray(state_rows, dtype=np.float32)
    mask = np.asarray(mask_rows, dtype=bool)
    previous = np.asarray(previous_rows, dtype=np.float32)
    task_one_hot = np.asarray(task_rows, dtype=np.float32)
    actions = np.asarray(action_rows, dtype=np.float32)
    if state.shape[1] != len(FEATURE_NAMES) or actions.shape[1] != OUTPUT_DIM:
        raise AssertionError((state.shape, actions.shape))
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise RuntimeError("non-finite generator training value")
    atomic_npz(
        destination,
        schema_version=np.asarray(DATA_SCHEMA),
        observable_state=state,
        observable_mask=mask,
        previous_action=previous,
        task_one_hot=task_one_hot,
        target_action_chunk=actions,
        task_id=np.asarray(task_name_rows),
        episode_id=np.asarray(episode_rows, dtype=np.int16),
        timestep=np.asarray(timestep_rows, dtype=np.int16),
        feature_names=np.asarray(FEATURE_NAMES),
    )
    mark_complete(
        destination,
        {
            "kind": "stage5_generator_training_data",
            "schema_version": DATA_SCHEMA,
            "rows": int(len(state)),
            "episodes": list(GENERATOR_TRAIN_EPISODES),
            "tasks": [task["task_id"] for task in TASKS],
            "stage5_consequence_labels_read": False,
        },
    )
    return {
        "path": destination,
        "sha256": sha256_file(destination),
        "rows": int(len(state)),
        "reused": False,
    }


def _fit_scaler(state, mask, previous):
    center = np.zeros(state.shape[1], dtype=np.float32)
    scale = np.ones(state.shape[1], dtype=np.float32)
    for column in range(state.shape[1]):
        values = state[mask[:, column], column]
        if len(values):
            center[column] = float(np.mean(values))
            scale[column] = max(float(np.std(values)), 1e-3)
    previous_center = np.mean(previous, axis=0).astype(np.float32)
    previous_scale = np.maximum(np.std(previous, axis=0), 1e-3).astype(np.float32)
    return center, scale, previous_center, previous_scale


def assemble_input(
    state,
    mask,
    previous,
    task_one_hot,
    state_center,
    state_scale,
    previous_center,
    previous_scale,
):
    state = np.asarray(state, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    normalized_state = (state - state_center) / state_scale
    normalized_state = np.where(mask, normalized_state, 0.0)
    normalized_previous = (np.asarray(previous, dtype=np.float32) - previous_center) / previous_scale
    return np.concatenate(
        (
            normalized_state,
            mask.astype(np.float32),
            normalized_previous,
            np.asarray(task_one_hot, dtype=np.float32),
        ),
        axis=-1,
    ).astype(np.float32)


def train(project_root, output_root=None, scratch_root=SCRATCH_ROOT):
    import torch
    from torch import nn

    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    data_path = training_data_path(scratch_root)
    data = _load_complete_npz(data_path, DATA_SCHEMA)
    state_center, state_scale, previous_center, previous_scale = _fit_scaler(
        data["observable_state"], data["observable_mask"], data["previous_action"]
    )
    x = assemble_input(
        data["observable_state"],
        data["observable_mask"],
        data["previous_action"],
        data["task_one_hot"],
        state_center,
        state_scale,
        previous_center,
        previous_scale,
    )
    y = np.asarray(data["target_action_chunk"], dtype=np.float32)
    if x.shape[1] != INPUT_DIM or y.shape[1] != OUTPUT_DIM:
        raise AssertionError((x.shape, y.shape))
    torch.set_num_threads(min(16, max(1, os.cpu_count() or 1)))
    torch.manual_seed(GENERATOR_TRAIN_SEED)
    np.random.seed(GENERATOR_TRAIN_SEED)
    torch.use_deterministic_algorithms(True)
    model = nn.Sequential(
        nn.Linear(INPUT_DIM, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, OUTPUT_DIM),
        nn.Tanh(),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(GENERATOR_ARCHITECTURE["learning_rate"]),
        weight_decay=float(GENERATOR_ARCHITECTURE["weight_decay"]),
    )
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    weights = torch.ones(OUTPUT_DIM, dtype=torch.float32)
    weights[np.arange(config.ACTION_DIM - 1, OUTPUT_DIM, config.ACTION_DIM)] = float(
        GENERATOR_ARCHITECTURE["gripper_loss_weight"]
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(GENERATOR_TRAIN_SEED)
    trace = []
    started = time.time()
    model.train()
    for step in range(1, int(GENERATOR_ARCHITECTURE["steps"]) + 1):
        indices = torch.randint(
            len(x_tensor),
            (int(GENERATOR_ARCHITECTURE["batch_size"]),),
            generator=generator,
        )
        prediction = model(x_tensor[indices])
        loss = torch.mean((prediction - y_tensor[indices]) ** 2 * weights[None, :])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 250 == 0:
            value = float(loss.detach().cpu())
            trace.append({"step": int(step), "weighted_mse": value})
            print(
                "generator-train step=%d loss=%.8f elapsed=%.1fs"
                % (step, value, time.time() - started),
                flush=True,
            )
    model.eval()
    with torch.no_grad():
        prediction = []
        for start in range(0, len(x_tensor), 2048):
            prediction.append(model(x_tensor[start : start + 2048]).cpu().numpy())
    prediction = np.concatenate(prediction, axis=0)
    error = prediction - y
    continuous_columns = np.asarray(
        [index for index in range(OUTPUT_DIM) if (index + 1) % config.ACTION_DIM != 0],
        dtype=np.int64,
    )
    gripper_columns = np.asarray(
        list(range(config.ACTION_DIM - 1, OUTPUT_DIM, config.ACTION_DIM)),
        dtype=np.int64,
    )
    metrics = {
        "training_rows": int(len(x)),
        "full_rmse": float(np.sqrt(np.mean(error ** 2))),
        "continuous_rmse": float(np.sqrt(np.mean(error[:, continuous_columns] ** 2))),
        "gripper_rmse": float(np.sqrt(np.mean(error[:, gripper_columns] ** 2))),
        "trace": trace,
        "selection_or_early_stopping": False,
        "development_data_read": False,
    }
    linear_layers = [layer for layer in model if isinstance(layer, nn.Linear)]
    arrays = {
        "schema_version": np.asarray(CHECKPOINT_SCHEMA),
        "state_center": state_center,
        "state_scale": state_scale,
        "previous_center": previous_center,
        "previous_scale": previous_scale,
        "training_data_sha256": np.asarray(sha256_file(data_path)),
    }
    for index, layer in enumerate(linear_layers):
        arrays["weight_%d" % index] = layer.weight.detach().cpu().numpy().astype(np.float32)
        arrays["bias_%d" % index] = layer.bias.detach().cpu().numpy().astype(np.float32)
    destination = checkpoint_path(output_root)
    atomic_npz(destination, **arrays)
    mark_complete(
        destination,
        {
            "kind": "stage5_state_h4_bc_checkpoint",
            "schema_version": CHECKPOINT_SCHEMA,
            "training_data_sha256": sha256_file(data_path),
            "development_data_read": False,
        },
    )
    checkpoint_hash = sha256_file(destination)
    metrics_path = os.path.join(output_root, "NOMINAL_GENERATOR_TRAINING.json")
    atomic_json(
        metrics_path,
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "architecture": GENERATOR_ARCHITECTURE,
            "checkpoint_path": os.path.relpath(destination, output_root),
            "checkpoint_sha256": checkpoint_hash,
            "training_data_path": data_path,
            "training_data_sha256": sha256_file(data_path),
            "metrics": metrics,
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "device": "cpu",
                "torch_threads": torch.get_num_threads(),
            },
        },
    )
    binding_path = os.path.join(output_root, "NOMINAL_GENERATOR_BINDING.json")
    with open(binding_path, "r", encoding="utf-8") as handle:
        binding = json.load(handle)
    if binding.get("checkpoint_status") != "PENDING_TRAINING_BEFORE_DEVELOPMENT":
        raise RuntimeError("generator binding was already frozen")
    binding.update(
        {
            "checkpoint_status": "FROZEN_BEFORE_DEVELOPMENT",
            "checkpoint_path": os.path.relpath(destination, project_root),
            "checkpoint_sha256": checkpoint_hash,
            "training_data_path": data_path,
            "training_data_sha256": sha256_file(data_path),
            "training_rows": int(len(x)),
            "development_data_read": False,
            "training_metrics_artifact": "NOMINAL_GENERATOR_TRAINING.json",
            "training_metrics_sha256": sha256_file(metrics_path),
        }
    )
    atomic_json(binding_path, binding)
    return binding


def load_numpy_generator(path):
    values = _load_complete_npz(path, CHECKPOINT_SCHEMA)
    layers = []
    index = 0
    while "weight_%d" % index in values:
        layers.append((values["weight_%d" % index], values["bias_%d" % index]))
        index += 1
    if len(layers) != 4:
        raise RuntimeError("unexpected generator layer count")
    return values, layers


def predict_numpy(checkpoint, state, mask, previous, task_one_hot):
    values, layers = checkpoint
    x = assemble_input(
        state,
        mask,
        previous,
        task_one_hot,
        values["state_center"],
        values["state_scale"],
        values["previous_center"],
        values["previous_scale"],
    )
    for index, (weight, bias) in enumerate(layers):
        x = np.matmul(x, weight.T) + bias
        x = np.tanh(x) if index == len(layers) - 1 else np.maximum(x, 0.0)
    return np.asarray(x, dtype=np.float32).reshape(x.shape[:-1] + (config.CHUNK_HORIZON, config.ACTION_DIM))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("collect", "train"))
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--libero-source", default=config.LIBERO_SOURCE_DEFAULT)
    parser.add_argument("--dataset-root", default=config.DATASET_ROOT_DEFAULT)
    parser.add_argument("--scratch-root", default=SCRATCH_ROOT)
    args = parser.parse_args(argv)
    if args.command == "collect":
        result = collect_training_data(args.libero_source, args.dataset_root, args.scratch_root)
    else:
        result = train(args.project_root, scratch_root=args.scratch_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
