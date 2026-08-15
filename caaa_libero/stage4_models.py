"""Context-reversal consequence embedding models and matched controls.

These are small state/action metric models.  They never receive future
consequences at inference and are not robot policies.
"""

from __future__ import annotations

import copy
import json
import os
import time

import numpy as np

from .stage3_data import CONTEXT_SLICES
from .stage4_config import (
    ACTION_BANK_SIZE,
    ACTION_HIDDEN,
    CONTEXT_HIDDEN,
    CR_BATCH_SIZE,
    CR_C3_SEEDS,
    CR_LEARNING_RATE,
    CR_LISTWISE_WEIGHT,
    CR_MAX_EPOCHS,
    CR_MODEL_FAMILIES,
    CR_PAIRWISE_WEIGHT,
    CR_REVERSAL_WEIGHT,
    CR_TAU_MODEL,
    CR_TAU_TRUE,
    CR_WEIGHT_DECAY,
    EMBEDDING_DIM,
    GROUP_EMBEDDING_DIM,
    OUTPUT_RELATIVE,
    PHASES,
    REVERSAL_PAIR_SEED,
    SCRATCH_ROOT,
    SUPPORT_TARGET_COUNT,
    TASK_IDS,
)
from .stage4_data import _cache_path, load_cache, reversal_pairs, train_robust_margins
from .storage import atomic_json, sha256_file


PROPOSED_CONTROL = "PROPOSED"
MATCHED_CONTROLS = (
    "ACTION_ONLY",
    "CONTEXT_SHUFFLED",
    "NOMINAL_SHUFFLED",
    "CONSEQUENCE_LABEL_SHUFFLED",
    "REVERSAL_LABEL_SHUFFLED",
    "NO_REVERSAL_LOSS",
)


def _torch():
    import torch

    return torch


def _make_mlp(input_dim, hidden, output_dim):
    import torch.nn as nn

    layers = []
    previous = int(input_dim)
    for width in hidden:
        layers.extend((nn.Linear(previous, int(width)), nn.GELU()))
        previous = int(width)
    layers.append(nn.Linear(previous, int(output_dim)))
    return nn.Sequential(*layers)


def create_cr_model(family, context_dim=321, residual_dim=24):
    torch = _torch()
    import torch.nn as nn

    if family not in CR_MODEL_FAMILIES:
        raise KeyError(family)
    groups = 1 if family == "CR_C3_SHARED" else 5
    embedding_dim = EMBEDDING_DIM if groups == 1 else GROUP_EMBEDDING_DIM

    class ContextReversalEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.context_network = _make_mlp(
                int(context_dim), CONTEXT_HIDDEN, int(CONTEXT_HIDDEN[-1])
            )
            self.action_network = _make_mlp(
                int(CONTEXT_HIDDEN[-1]) + int(residual_dim),
                ACTION_HIDDEN,
                groups * embedding_dim,
            )
            self.groups = groups
            self.embedding_dim = embedding_dim

        def embed(self, context, residual):
            latent = self.context_network(context)
            value = self.action_network(torch.cat((latent, residual), dim=-1))
            return value.reshape(-1, self.groups, self.embedding_dim)

        def pair_distance(self, context, target, candidate):
            target_embedding = self.embed(context, target)
            candidate_embedding = self.embed(context, candidate)
            return torch.mean(
                torch.sum((target_embedding - candidate_embedding) ** 2, dim=-1),
                dim=-1,
            )

        def matrix_distance(self, target_embedding, candidate_embedding):
            difference = (
                target_embedding[:, None, :, :] - candidate_embedding[None, :, :, :]
            )
            return torch.mean(torch.sum(difference ** 2, dim=-1), dim=-1)

    return ContextReversalEmbedding()


def parameter_count(model):
    return int(sum(value.numel() for value in model.parameters()))


def _device(name):
    torch = _torch()
    device = torch.device(name or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one local GPU for Stage 4")
    return device


def _seed(*parts):
    import hashlib

    value = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


def _set_determinism(seed, device):
    torch = _torch()
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _load_pair_arrays(path):
    import pandas as pd

    frame = pd.read_parquet(path)
    numeric = {
        name: frame[name].to_numpy()
        for name in (
            "state_s1",
            "state_s2",
            "target_id",
            "candidate_i",
            "candidate_j",
            "margin",
        )
    }
    numeric["task_id"] = frame["task_id"].astype(str).to_numpy()
    numeric["phase"] = frame["phase"].astype(str).to_numpy()
    return numeric


def _pairs_from_cache(cache):
    margins, _ = train_robust_margins(cache)
    rows = reversal_pairs(cache, margins)
    names = (
        "state_s1",
        "state_s2",
        "target_id",
        "candidate_i",
        "candidate_j",
        "margin",
        "task_id",
        "phase",
    )
    return {name: np.asarray([row[name] for row in rows]) for name in names}


def _permutation(cache, seed, suffix):
    order = np.arange(len(cache["context"]), dtype=np.int64)
    for task in TASK_IDS:
        for phase in PHASES:
            keep = np.flatnonzero(
                (cache["task_id"].astype(str) == task)
                & (cache["phase"].astype(str) == phase)
            )
            rng = np.random.RandomState(_seed(seed, suffix, task, phase))
            order[keep] = rng.permutation(keep)
    return order


def controlled_training_data(cache, base_pairs, control, seed):
    """Apply a frozen matched control without changing model parameters/budget."""
    contexts = np.asarray(cache["context"], dtype=np.float32).copy()
    true_distance = np.asarray(cache["true_distance"], dtype=np.float32)
    pairs = {name: np.asarray(value).copy() for name, value in base_pairs.items()}
    if control == PROPOSED_CONTROL or control == "NO_REVERSAL_LOSS":
        pass
    elif control == "ACTION_ONLY":
        contexts[:] = 0.0
    elif control == "CONTEXT_SHUFFLED":
        order = _permutation(cache, seed, control)
        contexts = contexts[order]
    elif control == "NOMINAL_SHUFFLED":
        order = _permutation(cache, seed, control)
        left, right = CONTEXT_SLICES["nominal_action"]
        contexts[:, left:right] = contexts[order, left:right]
    elif control == "CONSEQUENCE_LABEL_SHUFFLED":
        order = _permutation(cache, seed, control)
        true_distance = true_distance[order].copy()
        changed = dict(cache)
        changed["true_distance"] = true_distance
        pairs = _pairs_from_cache(changed)
    elif control == "REVERSAL_LABEL_SHUFFLED":
        # Keep the correct listwise/pairwise surface but break only the
        # cross-state reversal label association inside each task/phase.
        for task in TASK_IDS:
            for phase in PHASES:
                keep = np.flatnonzero(
                    (pairs["task_id"].astype(str) == task)
                    & (pairs["phase"].astype(str) == phase)
                )
                rng = np.random.RandomState(_seed(seed, control, task, phase))
                permuted = rng.permutation(keep)
                for name in ("state_s2", "candidate_i", "candidate_j"):
                    pairs[name][keep] = pairs[name][permuted]
                collision = pairs["candidate_i"][keep] == pairs["candidate_j"][keep]
                pairs["candidate_j"][keep[collision]] = (
                    pairs["candidate_j"][keep[collision]] + 1
                ) % ACTION_BANK_SIZE
    else:
        raise KeyError(control)
    return contexts, true_distance, pairs


def _matrix_scores(model, context, targets, candidates, device):
    torch = _torch()
    actions = np.concatenate((targets, candidates), axis=0).astype(np.float32) / 0.12
    repeated_context = np.repeat(
        np.asarray(context, dtype=np.float32)[None, :], len(actions), axis=0
    )
    embedding = model.embed(
        torch.as_tensor(repeated_context, device=device),
        torch.as_tensor(actions, device=device),
    )
    target_embedding = embedding[: len(targets)]
    candidate_embedding = embedding[len(targets) :]
    return model.matrix_distance(target_embedding, candidate_embedding)


def _ranking_losses(scores, true_distance):
    torch = _torch()
    true_probability = torch.softmax(-true_distance / float(CR_TAU_TRUE), dim=1)
    model_log_probability = torch.log_softmax(-scores / float(CR_TAU_MODEL), dim=1)
    listwise = -torch.mean(torch.sum(true_probability * model_log_probability, dim=1))
    true_order = torch.argsort(true_distance, dim=1, stable=True)
    positive_ids = true_order[:, :8]
    negative_ids = true_order[:, 8:]
    positives = torch.gather(scores, 1, positive_ids)
    negatives = torch.gather(scores, 1, negative_ids)
    pairwise = torch.mean(
        torch.nn.functional.softplus(
            positives[:, :, None] - negatives[:, None, :]
        )
    )
    return listwise, pairwise


def _reversal_loss(model, contexts, targets, candidates, pairs, ids, device):
    torch = _torch()
    s1 = np.asarray(pairs["state_s1"][ids], dtype=np.int64)
    s2 = np.asarray(pairs["state_s2"][ids], dtype=np.int64)
    target_id = np.asarray(pairs["target_id"][ids], dtype=np.int64)
    candidate_i = np.asarray(pairs["candidate_i"][ids], dtype=np.int64)
    candidate_j = np.asarray(pairs["candidate_j"][ids], dtype=np.int64)
    target = torch.as_tensor(targets[target_id] / 0.12, device=device)
    action_i = torch.as_tensor(candidates[candidate_i] / 0.12, device=device)
    action_j = torch.as_tensor(candidates[candidate_j] / 0.12, device=device)
    context_s1 = torch.as_tensor(contexts[s1], device=device)
    context_s2 = torch.as_tensor(contexts[s2], device=device)
    d_s1_i = model.pair_distance(context_s1, target, action_i)
    d_s1_j = model.pair_distance(context_s1, target, action_j)
    d_s2_i = model.pair_distance(context_s2, target, action_i)
    d_s2_j = model.pair_distance(context_s2, target, action_j)
    margin = torch.as_tensor(np.asarray(pairs["margin"][ids], dtype=np.float32), device=device)
    return torch.mean(
        torch.nn.functional.softplus(margin + d_s1_i - d_s1_j)
        + torch.nn.functional.softplus(margin + d_s2_j - d_s2_i)
    )


def train_model(cache, base_pairs, family, control, seed, device):
    torch = _torch()
    _set_determinism(seed, device)
    model = create_cr_model(family, context_dim=cache["context"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CR_LEARNING_RATE, weight_decay=CR_WEIGHT_DECAY
    )
    contexts, true_distance, pairs = controlled_training_data(
        cache, base_pairs, control, seed
    )
    targets = np.asarray(cache["target_residual"], dtype=np.float32)
    candidates = np.asarray(cache["candidate_residual"], dtype=np.float32)
    trace = []
    global_step = 0
    started = time.perf_counter()
    for epoch in range(CR_MAX_EPOCHS):
        model.train()
        state_order = np.random.RandomState(_seed(seed, control, epoch, "state")).permutation(
            len(contexts)
        )
        pair_order = np.random.RandomState(_seed(seed, control, epoch, "reversal")).permutation(
            len(pairs["state_s1"])
        )
        pair_cursor = 0
        epoch_parts = []
        for state in state_order:
            target_order = np.random.RandomState(
                _seed(seed, control, epoch, int(state), "target")
            ).permutation(SUPPORT_TARGET_COUNT)
            for start in range(0, SUPPORT_TARGET_COUNT, CR_BATCH_SIZE):
                query_ids = target_order[start : start + CR_BATCH_SIZE]
                optimizer.zero_grad(set_to_none=True)
                score = _matrix_scores(
                    model, contexts[state], targets[query_ids], candidates, device
                )
                truth = torch.as_tensor(
                    true_distance[state, query_ids], device=device
                )
                listwise, pairwise = _ranking_losses(score, truth)
                reversal_ids = np.take(
                    pair_order,
                    np.arange(pair_cursor, pair_cursor + CR_BATCH_SIZE)
                    % len(pair_order),
                )
                pair_cursor = (pair_cursor + CR_BATCH_SIZE) % len(pair_order)
                if control == "NO_REVERSAL_LOSS":
                    reversal = torch.zeros((), device=device)
                    reversal_weight = 0.0
                else:
                    reversal = _reversal_loss(
                        model,
                        contexts,
                        targets,
                        candidates,
                        pairs,
                        reversal_ids,
                        device,
                    )
                    reversal_weight = CR_REVERSAL_WEIGHT
                total = (
                    CR_LISTWISE_WEIGHT * listwise
                    + CR_PAIRWISE_WEIGHT * pairwise
                    + reversal_weight * reversal
                )
                total.backward()
                optimizer.step()
                epoch_parts.append(
                    (
                        float(total.detach().cpu()),
                        float(listwise.detach().cpu()),
                        float(pairwise.detach().cpu()),
                        float(reversal.detach().cpu()),
                    )
                )
                global_step += 1
        mean = np.mean(np.asarray(epoch_parts, dtype=np.float64), axis=0)
        trace.append(
            {
                "epoch": epoch,
                "total": float(mean[0]),
                "listwise": float(mean[1]),
                "pairwise": float(mean[2]),
                "reversal": float(mean[3]),
                "optimizer_steps": len(epoch_parts),
            }
        )
        print(
            "CR_EPOCH family=%s control=%s seed=%d epoch=%d/%d loss=%.6f"
            % (family, control, seed, epoch + 1, CR_MAX_EPOCHS, mean[0]),
            flush=True,
        )
    model.eval()
    metadata = {
        "family": family,
        "control": control,
        "seed": int(seed),
        "epochs": CR_MAX_EPOCHS,
        "query_batch_size": CR_BATCH_SIZE,
        "queries_per_epoch": len(contexts) * SUPPORT_TARGET_COUNT,
        "full_bank_candidates_per_query": ACTION_BANK_SIZE,
        "optimizer_steps": global_step,
        "parameter_count": parameter_count(model),
        "wall_seconds": float(time.perf_counter() - started),
        "loss_weights": {
            "listwise": CR_LISTWISE_WEIGHT,
            "pairwise": CR_PAIRWISE_WEIGHT,
            "reversal": 0.0 if control == "NO_REVERSAL_LOSS" else CR_REVERSAL_WEIGHT,
        },
        "trace": trace,
    }
    return model, metadata


def save_checkpoint(path, model, metadata):
    torch = _torch()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    torch.save(
        {
            "family": metadata["family"],
            "control": metadata["control"],
            "seed": metadata["seed"],
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        temporary,
    )
    os.replace(temporary, path)
    evidence = copy.deepcopy(metadata)
    evidence["checkpoint_sha256"] = sha256_file(path)
    atomic_json(path + ".json", evidence)
    return evidence


def load_checkpoint(path, device):
    torch = _torch()
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    model = create_cr_model(payload["family"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["metadata"]


def score_cache(models, cache, device):
    """Return ensemble mean and per-member [state,target,candidate] distances."""
    members = []
    targets = np.asarray(cache["target_residual"], dtype=np.float32)
    candidates = np.asarray(cache["candidate_residual"], dtype=np.float32)
    torch = _torch()
    for member_index, model in enumerate(models):
        values = np.empty(
            (len(cache["context"]), SUPPORT_TARGET_COUNT, ACTION_BANK_SIZE),
            dtype=np.float32,
        )
        model.eval()
        with torch.no_grad():
            for state, context in enumerate(cache["context"]):
                values[state] = _matrix_scores(
                    model, context, targets, candidates, device
                ).cpu().numpy()
        members.append(values)
    member = np.stack(members)
    return np.mean(member, axis=0), member


def ensemble_action_embedding(models, context, residuals, device):
    """Euclidean concatenation exactly representing mean member/group distance."""
    torch = _torch()
    residuals = np.asarray(residuals, dtype=np.float32)
    repeated = np.repeat(
        np.asarray(context, dtype=np.float32)[None, :], len(residuals), axis=0
    )
    values = []
    with torch.no_grad():
        for model in models:
            embedded = model.embed(
                torch.as_tensor(repeated, device=device),
                torch.as_tensor(residuals / 0.12, device=device),
            )
            values.append(embedded.reshape(len(residuals), -1).cpu().numpy())
    groups = int(models[0].groups)
    return np.concatenate(values, axis=1) / np.sqrt(float(len(models) * groups))


def evaluation_cache_for_control(cache, control):
    if control != "ACTION_ONLY":
        return cache
    changed = dict(cache)
    changed["context"] = np.zeros_like(cache["context"], dtype=np.float32)
    return changed


def ranking_summary(true_distance, predicted):
    from scipy.stats import rankdata

    true_distance = np.asarray(true_distance, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    selected = np.argmin(predicted, axis=2)
    oracle = np.min(true_distance, axis=2)
    chosen = np.take_along_axis(true_distance, selected[..., None], axis=2)[..., 0]
    true_rank = rankdata(true_distance, axis=2, method="average")
    pred_rank = rankdata(predicted, axis=2, method="average")
    true_rank -= np.mean(true_rank, axis=2, keepdims=True)
    pred_rank -= np.mean(pred_rank, axis=2, keepdims=True)
    spearman = np.sum(true_rank * pred_rank, axis=2) / np.maximum(
        np.sqrt(
            np.sum(true_rank ** 2, axis=2) * np.sum(pred_rank ** 2, axis=2)
        ),
        1e-12,
    )
    relevance = np.exp(-true_distance / float(CR_TAU_TRUE))
    true_order = np.argsort(true_distance, axis=2, kind="stable")[:, :, :16]
    pred_order = np.argsort(predicted, axis=2, kind="stable")[:, :, :16]
    discount = 1.0 / np.log2(np.arange(2, 18))
    actual = np.sum(np.take_along_axis(relevance, pred_order, axis=2) * discount, axis=2)
    ideal = np.sum(np.take_along_axis(relevance, true_order, axis=2) * discount, axis=2)
    return {
        "mean_oracle_regret": float(np.mean(chosen - oracle)),
        "mean_spearman": float(np.mean(spearman)),
        "mean_ndcg_at_16": float(np.mean(actual / np.maximum(ideal, 1e-12))),
        "selected": selected,
    }


def reversal_accuracy(predicted, pairs):
    s1 = np.asarray(pairs["state_s1"], dtype=np.int64)
    s2 = np.asarray(pairs["state_s2"], dtype=np.int64)
    target = np.asarray(pairs["target_id"], dtype=np.int64)
    candidate_i = np.asarray(pairs["candidate_i"], dtype=np.int64)
    candidate_j = np.asarray(pairs["candidate_j"], dtype=np.int64)
    correct_s1 = predicted[s1, target, candidate_i] < predicted[s1, target, candidate_j]
    correct_s2 = predicted[s2, target, candidate_j] < predicted[s2, target, candidate_i]
    return {
        "joint_context_reversal_accuracy": float(np.mean(correct_s1 & correct_s2)),
        "side_context_reversal_accuracy": float(np.mean(np.concatenate((correct_s1, correct_s2)))),
        "pair_count": len(s1),
    }


def _model_path(output_root, family, control, member_index):
    return os.path.join(
        output_root,
        "models",
        "cr_c3",
        "%s__%s__member_%d.pt" % (family, control, int(member_index)),
    )


def _calibration_pairs(cache, train_pairs_path):
    # Margins remain train-only.  Generate calibration examples with those
    # frozen margins and never use calibration to fit a margin.
    import pandas as pd

    train_frame = pd.read_parquet(train_pairs_path)
    margins = {
        (task, phase): float(
            train_frame[
                (train_frame.task_id == task) & (train_frame.phase == phase)
            ].margin.iloc[0]
        )
        for task in TASK_IDS
        for phase in PHASES
    }
    return {
        name: np.asarray([row[name] for row in reversal_pairs(cache, margins)])
        for name in (
            "state_s1",
            "state_s2",
            "target_id",
            "candidate_i",
            "candidate_j",
            "margin",
            "task_id",
            "phase",
        )
    }


def train_proposed_families(project_root, output_root=None, device_name=None):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    device = _device(device_name)
    train_cache = load_cache(_cache_path(SCRATCH_ROOT, "train_matrix_cache"))
    calibration_cache = load_cache(
        _cache_path(SCRATCH_ROOT, "historical_calibration_matrix_cache")
    )
    train_pairs_path = os.path.join(output_root, "CONTEXT_REVERSAL_PAIRS.parquet")
    train_pairs = _load_pair_arrays(train_pairs_path)
    calibration_pairs = _calibration_pairs(calibration_cache, train_pairs_path)
    family_results = []
    for family in CR_MODEL_FAMILIES:
        models = []
        checkpoints = []
        for member_index, seed in enumerate(CR_C3_SEEDS):
            path = _model_path(output_root, family, PROPOSED_CONTROL, member_index)
            if os.path.isfile(path) and os.path.isfile(path + ".json"):
                model, metadata = load_checkpoint(path, device)
                evidence = json.load(open(path + ".json", "r", encoding="utf-8"))
                if evidence["checkpoint_sha256"] != sha256_file(path):
                    raise RuntimeError("checkpoint hash mismatch: " + path)
            else:
                model, metadata = train_model(
                    train_cache, train_pairs, family, PROPOSED_CONTROL, seed, device
                )
                evidence = save_checkpoint(path, model, metadata)
            models.append(model)
            checkpoints.append(
                {
                    "member_index": member_index,
                    "seed": seed,
                    "path": os.path.relpath(path, output_root),
                    "sha256": sha256_file(path),
                    "parameter_count": parameter_count(model),
                }
            )
        predicted, per_member = score_cache(models, calibration_cache, device)
        ranking = ranking_summary(calibration_cache["true_distance"], predicted)
        reversal = reversal_accuracy(predicted, calibration_pairs)
        member_metrics = [
            {
                **ranking_summary(calibration_cache["true_distance"], values),
                **reversal_accuracy(values, calibration_pairs),
            }
            for values in per_member
        ]
        for value in member_metrics:
            value.pop("selected", None)
        ranking.pop("selected", None)
        family_results.append(
            {
                "family": family,
                "checkpoints": checkpoints,
                "calibration": {**ranking, **reversal},
                "member_calibration": member_metrics,
            }
        )
    selected_index = min(
        range(len(family_results)),
        key=lambda index: (
            family_results[index]["calibration"]["mean_oracle_regret"],
            -family_results[index]["calibration"]["joint_context_reversal_accuracy"],
            -family_results[index]["calibration"]["mean_ndcg_at_16"],
            index,
        ),
    )
    path = os.path.join(output_root, "MODEL_SELECTION.json")
    selection = json.load(open(path, "r", encoding="utf-8"))
    selection["cr_c3_selection"] = {
        "split": "calibration episodes 32-35 only",
        "family_trace": family_results,
        "selection_order": [
            "lowest calibration full-bank oracle regret",
            "highest calibration joint context-reversal accuracy",
            "highest calibration NDCG@16",
            "lowest frozen family index",
        ],
        "selected_family_index": selected_index,
        "selected_family": family_results[selected_index]["family"],
        "development_or_historical_used": False,
    }
    atomic_json(path, selection)
    return selection["cr_c3_selection"]


def train_matched_controls(project_root, output_root=None, device_name=None):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    device = _device(device_name)
    selection_path = os.path.join(output_root, "MODEL_SELECTION.json")
    selection = json.load(open(selection_path, "r", encoding="utf-8"))
    family = selection["cr_c3_selection"]["selected_family"]
    train_cache = load_cache(_cache_path(SCRATCH_ROOT, "train_matrix_cache"))
    calibration_cache = load_cache(
        _cache_path(SCRATCH_ROOT, "historical_calibration_matrix_cache")
    )
    train_pairs_path = os.path.join(output_root, "CONTEXT_REVERSAL_PAIRS.parquet")
    train_pairs = _load_pair_arrays(train_pairs_path)
    calibration_pairs = _calibration_pairs(calibration_cache, train_pairs_path)
    results = []
    for control in MATCHED_CONTROLS:
        models = []
        checkpoints = []
        for member_index, seed in enumerate(CR_C3_SEEDS):
            path = _model_path(output_root, family, control, member_index)
            if os.path.isfile(path) and os.path.isfile(path + ".json"):
                model, metadata = load_checkpoint(path, device)
            else:
                model, metadata = train_model(
                    train_cache, train_pairs, family, control, seed, device
                )
                save_checkpoint(path, model, metadata)
            models.append(model)
            checkpoints.append(
                {
                    "member_index": member_index,
                    "seed": seed,
                    "path": os.path.relpath(path, output_root),
                    "sha256": sha256_file(path),
                    "parameter_count": parameter_count(model),
                }
            )
        predicted, per_member = score_cache(
            models, evaluation_cache_for_control(calibration_cache, control), device
        )
        ranking = ranking_summary(calibration_cache["true_distance"], predicted)
        ranking.pop("selected", None)
        result = {
            "control": control,
            "family": family,
            "checkpoints": checkpoints,
            "calibration": {
                **ranking,
                **reversal_accuracy(predicted, calibration_pairs),
            },
        }
        results.append(result)
    expected = selection["cr_c3_selection"]["family_trace"][
        selection["cr_c3_selection"]["selected_family_index"]
    ]["checkpoints"][0]["parameter_count"]
    mismatched = [
        row
        for result in results
        for row in result["checkpoints"]
        if int(row["parameter_count"]) != int(expected)
    ]
    if mismatched:
        raise RuntimeError("matched control parameter-count mismatch")
    selection["cr_c3_controls"] = {
        "selected_family": family,
        "matched_parameter_count": expected,
        "same_seeds_architecture_ensemble_and_training_budget": True,
        "controls": results,
    }
    atomic_json(selection_path, selection)
    return selection["cr_c3_controls"]


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train-families", "train-controls"))
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if args.command == "train-families":
        result = train_proposed_families(
            args.project_root, args.output_root, args.device
        )
    else:
        result = train_matched_controls(
            args.project_root, args.output_root, args.device
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
