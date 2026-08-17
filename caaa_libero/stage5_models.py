"""Stage 5 static and context-gated positive-semidefinite action metrics.

The proposed model has no additive context path.  Its action representation
and positive base weights are copied from, then frozen with, B2.  Only the
bounded multiplicative context modulator is optimized for P1 and its matched
controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from .stage3_data import CONTEXT_SLICES, HISTORY_CONTROL_SLICES, STATE_CONTROL_SLICES
from .stage3_metrics import ranking_metrics
from .stage5_config import (
    ACTION_ENCODER_HIDDEN,
    CONTEXT_MODULATOR_HIDDEN,
    EMBEDDING_DIM,
    GRADIENT_CLIP_NORM,
    HUBER_DELTA,
    LEARNING_RATE,
    LOCAL_BANK_SIZE,
    LOSS_WEIGHTS,
    MATCHED_CONTROLS,
    MODEL_SEEDS,
    MODULATION_LOG_BOUND,
    OUTPUT_RELATIVE,
    REVERSAL_BATCH,
    SCRATCH_ROOT,
    TEMPERATURE_CANDIDATES,
    TRAINING_STEPS,
    TRAIN_QUERY_BATCH,
    WEIGHT_DECAY,
)
from .stage5_data import cache_path, load_cache
from .storage import atomic_json, sha256_file


PROPOSED = "PROPOSED"
BASE_METHODS = ("B1_ACTION_ONLY", "B2_STATIC_CONSEQUENCE")
CONTEXT_METHOD = "P1_CONTEXT_GATED_PSD"


def _torch():
    import torch

    return torch


def _mlp(input_dim, hidden, output_dim):
    import torch.nn as nn

    layers = []
    previous = int(input_dim)
    for width in hidden:
        layers.extend((nn.Linear(previous, int(width)), nn.GELU()))
        previous = int(width)
    layers.append(nn.Linear(previous, int(output_dim)))
    return nn.Sequential(*layers)


class ActionEncoder:
    """Factory namespace so torch remains an optional analysis dependency."""

    @staticmethod
    def create():
        return _mlp(48, ACTION_ENCODER_HIDDEN, EMBEDDING_DIM)


def create_static_metric():
    torch = _torch()
    import torch.nn as nn

    class StaticMetric(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = ActionEncoder.create()
            self.raw_weight = nn.Parameter(torch.zeros(EMBEDDING_DIM))

        def encode(self, nominal, residual):
            return self.encoder(torch.cat((nominal, residual / 0.12), dim=-1))

        def positive_weight(self):
            return torch.nn.functional.softplus(self.raw_weight) + 1e-8

        def distance(self, nominal, target, candidate):
            z_target = self.encode(nominal, target)
            z_candidate = self.encode(nominal, candidate)
            return torch.sum(
                self.positive_weight() * (z_target - z_candidate) ** 2, dim=-1
            )

        def matrix_distance(self, nominal, targets, candidates):
            # nominal: Bx24, targets: BxTx24, candidates: BxCx24
            batch, target_count = targets.shape[:2]
            candidate_count = candidates.shape[1]
            target_nominal = nominal[:, None, :].expand(-1, target_count, -1)
            candidate_nominal = nominal[:, None, :].expand(-1, candidate_count, -1)
            target_z = self.encode(target_nominal, targets)
            candidate_z = self.encode(candidate_nominal, candidates)
            difference = target_z[:, :, None, :] - candidate_z[:, None, :, :]
            return torch.sum(
                self.positive_weight()[None, None, None, :] * difference ** 2,
                dim=-1,
            )

    return StaticMetric()


def create_context_metric(base_model, context_dim=321):
    torch = _torch()
    import torch.nn as nn

    class ContextMetric(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)
            self.modulator = _mlp(
                int(context_dim), CONTEXT_MODULATOR_HIDDEN, EMBEDDING_DIM
            )
            self.register_buffer("context_offset", torch.zeros(EMBEDDING_DIM))

        def modulation(self, context):
            return float(MODULATION_LOG_BOUND) * torch.tanh(
                self.modulator(context) - self.context_offset
            )

        def positive_weight(self, context, force_zero_modulation=False):
            base = self.base.positive_weight()
            if force_zero_modulation:
                shape = context.shape[:-1] + (EMBEDDING_DIM,)
                return base.expand(shape)
            return base * torch.exp(self.modulation(context))

        def distance(
            self, context, nominal, target, candidate, force_zero_modulation=False
        ):
            target_z = self.base.encode(nominal, target)
            candidate_z = self.base.encode(nominal, candidate)
            weight = self.positive_weight(context, force_zero_modulation)
            return torch.sum(weight * (target_z - candidate_z) ** 2, dim=-1)

        def matrix_distance(
            self,
            context,
            nominal,
            targets,
            candidates,
            force_zero_modulation=False,
        ):
            batch, target_count = targets.shape[:2]
            candidate_count = candidates.shape[1]
            target_nominal = nominal[:, None, :].expand(-1, target_count, -1)
            candidate_nominal = nominal[:, None, :].expand(-1, candidate_count, -1)
            target_z = self.base.encode(target_nominal, targets)
            candidate_z = self.base.encode(candidate_nominal, candidates)
            difference = target_z[:, :, None, :] - candidate_z[:, None, :, :]
            weight = self.positive_weight(context, force_zero_modulation)
            return torch.sum(weight[:, None, None, :] * difference ** 2, dim=-1)

    return ContextMetric(base_model)


def parameter_count(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


def trainable_parameter_count(model):
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def _seed(*parts):
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _device(name):
    torch = _torch()
    selected = torch.device(name or "cpu")
    if selected.type == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if "," in visible or torch.cuda.device_count() != 1:
            raise RuntimeError("Stage 5 permits exactly one visible local GPU")
    return selected


def _set_determinism(seed, device):
    torch = _torch()
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _load_pairs(output_root):
    import pandas as pd

    frame = pd.read_parquet(os.path.join(output_root, "CONTEXT_REVERSAL_PAIRS.parquet"))
    frame = frame[frame["split"].astype(str) == "train"].reset_index(drop=True)
    names = (
        "state_s1",
        "state_s2",
        "target_id",
        "candidate_i",
        "candidate_j",
        "margin",
    )
    output = {name: frame[name].to_numpy() for name in names}
    if len(frame) < REVERSAL_BATCH:
        raise RuntimeError("too few frozen train reversals")
    return output


def _within_task_permutation(cache, seed, suffix):
    order = np.arange(len(cache["context"]), dtype=np.int64)
    for task in sorted(set(cache["task_id"].astype(str).tolist())):
        indices = np.flatnonzero(cache["task_id"].astype(str) == task)
        rng = np.random.RandomState(_seed(seed, suffix, task))
        order[indices] = rng.permutation(indices)
    return order


def controlled_training_arrays(cache, control, seed, action_truth):
    context = np.asarray(cache["context"], dtype=np.float32).copy()
    nominal = np.asarray(cache["nominal_action"], dtype=np.float32).copy()
    truth = np.asarray(cache["true_distance"], dtype=np.float32).copy()
    label_kind = "consequence"
    reversal_enabled = control not in ("ACTION_ONLY", "NO_REVERSAL_LOSS")
    if control in (PROPOSED, "NO_REVERSAL_LOSS"):
        pass
    elif control == "ACTION_ONLY":
        truth = np.broadcast_to(action_truth[None, :, :], truth.shape).copy()
        context[:] = 0.0
        label_kind = "action_only"
    elif control == "CONTEXT_SHUFFLED":
        order = _within_task_permutation(cache, seed, control)
        for name in STATE_CONTROL_SLICES + HISTORY_CONTROL_SLICES:
            left, right = CONTEXT_SLICES[name]
            context[:, left:right] = context[order, left:right]
    elif control == "NOMINAL_SHUFFLED":
        order = _within_task_permutation(cache, seed, control)
        left, right = CONTEXT_SLICES["nominal_action"]
        context[:, left:right] = context[order, left:right]
        nominal = nominal[order]
    elif control == "JOINT_STATE_NOMINAL_SHUFFLED":
        order = _within_task_permutation(cache, seed, control)
        keep_task = context[:, CONTEXT_SLICES["task_one_hot"][0] :].copy()
        context = context[order].copy()
        context[:, CONTEXT_SLICES["task_one_hot"][0] :] = keep_task
        nominal = nominal[order]
    elif control == "CONSEQUENCE_LABEL_SHUFFLED":
        order = _within_task_permutation(cache, seed, control)
        truth = truth[order].copy()
    elif control == "PHASE_ONLY":
        context[:] = 0.0
        context[np.arange(len(context)), np.asarray(cache["phase_index"], dtype=np.int64)] = 1.0
        left, right = CONTEXT_SLICES["task_one_hot"]
        context[:, left:right] = np.eye(right - left, dtype=np.float32)[
            np.asarray(cache["task_index"], dtype=np.int64)
        ]
    elif control == "CURRENT_CONTACT_ONLY":
        context[:] = 0.0
        left, right = CONTEXT_SLICES["current_contact"]
        context[:, left:right] = np.asarray(cache["current_contact"], dtype=np.float32)[:, None]
        left, right = CONTEXT_SLICES["task_one_hot"]
        context[:, left:right] = np.eye(right - left, dtype=np.float32)[
            np.asarray(cache["task_index"], dtype=np.int64)
        ]
    elif is_context:
        target = torch.as_tensor(cache["target_residual"][target_id], device=device)
        cand_i = torch.as_tensor(cache["candidate_residual"][candidate_i], device=device)
        cand_j = torch.as_tensor(cache["candidate_residual"][candidate_j], device=device)
        nominal1 = torch.as_tensor(arrays["nominal"][s1], device=device)
        nominal2 = torch.as_tensor(arrays["nominal"][s2], device=device)
        context1 = torch.as_tensor(arrays["context"][s1], device=device)
        context2 = torch.as_tensor(arrays["context"][s2], device=device)
        d1i = model.distance(context1, nominal1, target, cand_i)
        d1j = model.distance(context1, nominal1, target, cand_j)
        d2i = model.distance(context2, nominal2, target, cand_i)
        d2j = model.distance(context2, nominal2, target, cand_j)
    else:
        raise KeyError(control)
    return {
        "context": context,
        "nominal": nominal,
        "truth": truth,
        "label_kind": label_kind,
        "reversal_enabled": bool(reversal_enabled),
    }


def action_distance_matrix(project_root, output_root):
    with np.load(os.path.join(output_root, "LOCAL_BANK.npz"), allow_pickle=False) as data:
        whitener = np.asarray(data["train_covariance_whitener"], dtype=np.float64)
        candidates = np.asarray(data["residuals"], dtype=np.float64)
    train = load_cache(cache_path(SCRATCH_ROOT, "train"))
    targets = np.asarray(train["target_residual"], dtype=np.float64)
    difference = targets[:, None, :] - candidates[None, :, :]
    whitened = np.einsum("tcd,ed->tce", difference, whitener)
    distance = np.sqrt(np.mean(whitened ** 2, axis=-1))
    positive = distance[distance > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    return (distance / max(scale, 1e-12)).astype(np.float32)


def _pairwise_loss(scores, truth, action_truth, target_contact, candidate_contact):
    torch = _torch()
    batch, candidate_count = scores.shape
    true_order = torch.argsort(truth, dim=1, stable=True)
    action_order = torch.argsort(action_truth, dim=1, stable=True)
    groups = [true_order[:, :8], true_order[:, 8:32]]
    changed = target_contact[:, None] != candidate_contact
    changed_truth = torch.where(changed, truth, torch.full_like(truth, float("inf")))
    groups.append(torch.argsort(changed_truth, dim=1, stable=True)[:, :8])
    close_mask = torch.zeros_like(truth, dtype=torch.bool)
    close_mask.scatter_(1, action_order[:, : candidate_count // 2], True)
    close_far = torch.where(close_mask, truth, torch.full_like(truth, -float("inf")))
    groups.append(torch.argsort(close_far, dim=1, descending=True, stable=True)[:, :8])
    far_mask = ~close_mask
    far_close = torch.where(far_mask, truth, torch.full_like(truth, float("inf")))
    groups.append(torch.argsort(far_close, dim=1, stable=True)[:, :8])
    selected = torch.cat(groups, dim=1)
    selected_truth = torch.gather(truth, 1, selected)
    selected_scores = torch.gather(scores, 1, selected)
    order = torch.argsort(selected_truth, dim=1, stable=True)
    ranked_scores = torch.gather(selected_scores, 1, order)
    positives = ranked_scores[:, :8]
    negatives = ranked_scores[:, 8:]
    return torch.mean(torch.nn.functional.softplus(positives[:, :, None] - negatives[:, None, :]))


def _base_scores(model, nominal, targets, candidates):
    return model.matrix_distance(nominal, targets[:, None, :], candidates)[:, 0, :]


def _context_scores(model, context, nominal, targets, candidates):
    return model.matrix_distance(context, nominal, targets[:, None, :], candidates)[:, 0, :]


def _reversal_loss(model, arrays, cache, pairs, ids, device, is_context):
    torch = _torch()
    s1 = np.asarray(pairs["state_s1"][ids], dtype=np.int64)
    s2 = np.asarray(pairs["state_s2"][ids], dtype=np.int64)
    target_id = np.asarray(pairs["target_id"][ids], dtype=np.int64)
    candidate_i = np.asarray(pairs["candidate_i"][ids], dtype=np.int64)
    candidate_j = np.asarray(pairs["candidate_j"][ids], dtype=np.int64)
    if is_context and "target_embedding" in arrays:
        target_z1 = torch.as_tensor(arrays["target_embedding"][s1, target_id], device=device)
        target_z2 = torch.as_tensor(arrays["target_embedding"][s2, target_id], device=device)
        candidate_z1_i = torch.as_tensor(arrays["candidate_embedding"][s1, candidate_i], device=device)
        candidate_z1_j = torch.as_tensor(arrays["candidate_embedding"][s1, candidate_j], device=device)
        candidate_z2_i = torch.as_tensor(arrays["candidate_embedding"][s2, candidate_i], device=device)
        candidate_z2_j = torch.as_tensor(arrays["candidate_embedding"][s2, candidate_j], device=device)
        context1 = torch.as_tensor(arrays["context"][s1], device=device)
        context2 = torch.as_tensor(arrays["context"][s2], device=device)
        weight1 = model.positive_weight(context1)
        weight2 = model.positive_weight(context2)
        d1i = torch.sum(weight1 * (target_z1 - candidate_z1_i) ** 2, dim=-1)
        d1j = torch.sum(weight1 * (target_z1 - candidate_z1_j) ** 2, dim=-1)
        d2i = torch.sum(weight2 * (target_z2 - candidate_z2_i) ** 2, dim=-1)
        d2j = torch.sum(weight2 * (target_z2 - candidate_z2_j) ** 2, dim=-1)
    else:
        target = torch.as_tensor(cache["target_residual"][target_id], device=device)
        cand_i = torch.as_tensor(cache["candidate_residual"][candidate_i], device=device)
        cand_j = torch.as_tensor(cache["candidate_residual"][candidate_j], device=device)
        nominal1 = torch.as_tensor(arrays["nominal"][s1], device=device)
        nominal2 = torch.as_tensor(arrays["nominal"][s2], device=device)
        nominal_all = torch.cat((nominal1, nominal1, nominal1, nominal2, nominal2, nominal2), dim=0)
        residual_all = torch.cat((target, cand_i, cand_j, target, cand_i, cand_j), dim=0)
        encoded = model.encode(nominal_all, residual_all).reshape(6, len(ids), -1)
        weight = model.positive_weight()
        d1i = torch.sum(weight * (encoded[0] - encoded[1]) ** 2, dim=-1)
        d1j = torch.sum(weight * (encoded[0] - encoded[2]) ** 2, dim=-1)
        d2i = torch.sum(weight * (encoded[3] - encoded[4]) ** 2, dim=-1)
        d2j = torch.sum(weight * (encoded[3] - encoded[5]) ** 2, dim=-1)
    truth = arrays["truth"]
    desired1_i = truth[s1, target_id, candidate_i] < truth[s1, target_id, candidate_j]
    desired2_i = truth[s2, target_id, candidate_i] < truth[s2, target_id, candidate_j]
    sign1 = torch.as_tensor(np.where(desired1_i, 1.0, -1.0).astype(np.float32), device=device)
    sign2 = torch.as_tensor(np.where(desired2_i, 1.0, -1.0).astype(np.float32), device=device)
    margin = torch.as_tensor(np.asarray(pairs["margin"][ids], dtype=np.float32), device=device)
    return torch.mean(
        torch.nn.functional.softplus(margin + sign1 * (d1i - d1j))
        + torch.nn.functional.softplus(margin + sign2 * (d2i - d2j))
    )


def _gate_loss(model, context=None):
    torch = _torch()
    if context is None:
        weight = model.positive_weight()
        log_condition = torch.log(torch.max(weight) / torch.clamp(torch.min(weight), min=1e-8))
        return 1e-3 * log_condition ** 2
    modulation = model.modulation(context)
    weight = model.positive_weight(context)
    mean_penalty = torch.mean(torch.mean(modulation, dim=0) ** 2)
    norm_penalty = 1e-2 * torch.mean(modulation ** 2)
    condition = torch.max(weight, dim=1).values / torch.clamp(
        torch.min(weight, dim=1).values, min=1e-8
    )
    condition_penalty = 1e-3 * torch.mean(torch.log(condition) ** 2)
    return mean_penalty + norm_penalty + condition_penalty


def _train(
    model,
    arrays,
    cache,
    action_truth,
    pairs,
    tau,
    seed,
    device,
    is_context,
    label,
):
    torch = _torch()
    _set_determinism(seed, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.RandomState(_seed(seed, label, tau, "batches"))
    candidates_np = np.asarray(cache["candidate_residual"], dtype=np.float32)
    candidates = torch.as_tensor(candidates_np[None, :, :], device=device)
    action_truth_tensor = torch.as_tensor(action_truth, device=device)
    trace = []
    started = time.perf_counter()
    model.train()
    for step in range(1, TRAINING_STEPS + 1):
        states = rng.randint(len(arrays["context"]), size=TRAIN_QUERY_BATCH)
        targets = rng.randint(len(cache["target_residual"]), size=TRAIN_QUERY_BATCH)
        context = torch.as_tensor(arrays["context"][states], device=device)
        nominal = torch.as_tensor(arrays["nominal"][states], device=device)
        target = torch.as_tensor(cache["target_residual"][targets], device=device)
        expanded_candidates = candidates.expand(TRAIN_QUERY_BATCH, -1, -1)
        if is_context and "target_embedding" in arrays:
            target_z = torch.as_tensor(
                arrays["target_embedding"][states, targets], device=device
            )
            candidate_z = torch.as_tensor(
                arrays["candidate_embedding"][states], device=device
            )
            weight = model.positive_weight(context)
            score = torch.sum(
                weight[:, None, :] * (target_z[:, None, :] - candidate_z) ** 2,
                dim=-1,
            )
        else:
            score = (
                _context_scores(model, context, nominal, target, expanded_candidates)
                if is_context
                else _base_scores(model, nominal, target, expanded_candidates)
            )
        truth = torch.as_tensor(arrays["truth"][states, targets], device=device)
        query_action_truth = action_truth_tensor[targets]
        distance_loss = torch.nn.functional.huber_loss(
            torch.log1p(score), torch.log1p(truth), delta=HUBER_DELTA
        )
        pairwise_loss = _pairwise_loss(
            score,
            truth,
            query_action_truth,
            torch.as_tensor(cache["target_contact_mode"][states, targets], device=device),
            torch.as_tensor(cache["candidate_contact_mode"][states], device=device),
        )
        true_probability = torch.softmax(-truth / float(tau), dim=1)
        listwise_loss = -torch.mean(
            torch.sum(true_probability * torch.log_softmax(-score / float(tau), dim=1), dim=1)
        )
        if arrays["reversal_enabled"]:
            pair_ids = rng.randint(len(pairs["state_s1"]), size=REVERSAL_BATCH)
            reversal_loss = _reversal_loss(
                model, arrays, cache, pairs, pair_ids, device, is_context
            )
        else:
            reversal_loss = torch.zeros((), device=device)
        gate_loss = _gate_loss(model, context if is_context else None)
        loss = (
            LOSS_WEIGHTS["distance"] * distance_loss
            + LOSS_WEIGHTS["pairwise"] * pairwise_loss
            + LOSS_WEIGHTS["listwise"] * listwise_loss
            + LOSS_WEIGHTS["reversal"] * reversal_loss
            + LOSS_WEIGHTS["gate"] * gate_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP_NORM)
        optimizer.step()
        if step == 1 or step % 250 == 0:
            row = {
                "step": int(step),
                "total": float(loss.detach().cpu()),
                "distance": float(distance_loss.detach().cpu()),
                "pairwise": float(pairwise_loss.detach().cpu()),
                "listwise": float(listwise_loss.detach().cpu()),
                "reversal": float(reversal_loss.detach().cpu()),
                "gate": float(gate_loss.detach().cpu()),
            }
            trace.append(row)
            print(
                "stage5-train method=%s seed=%d tau=%.2f step=%d/%d loss=%.6f elapsed=%.1fs"
                % (label, seed, tau, step, TRAINING_STEPS, row["total"], time.perf_counter() - started),
                flush=True,
            )
    model.eval()
    return {
        "trace": trace,
        "optimizer_steps": TRAINING_STEPS,
        "elapsed_seconds": float(time.perf_counter() - started),
        "parameter_count": parameter_count(model),
        "trainable_parameter_count": trainable_parameter_count(model),
        "label_kind": arrays["label_kind"],
        "reversal_enabled": arrays["reversal_enabled"],
    }


def solve_zero_mean_offset(model, contexts, device):
    """Calibrate one monotone offset/dimension for exact train-mean zero."""
    torch = _torch()
    model.eval()
    raw = []
    with torch.no_grad():
        for start in range(0, len(contexts), 256):
            value = model.modulator(torch.as_tensor(contexts[start : start + 256], device=device))
            raw.append(value.detach().cpu().numpy())
    raw = np.concatenate(raw, axis=0).astype(np.float64)
    # The modulator is intentionally bounded after tanh, but its pre-tanh
    # logits need not lie in a fixed interval.  Bracket each monotone root from
    # the actually observed train logits; a fixed [-40, 40] bracket silently
    # fails when a normalized physical input drives a logit outside it.
    low = np.min(raw, axis=0) - 20.0
    high = np.max(raw, axis=0) + 20.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        mean = np.mean(np.tanh(raw - middle[None, :]), axis=0)
        low = np.where(mean > 0.0, middle, low)
        high = np.where(mean > 0.0, high, middle)
    offset = 0.5 * (low + high)
    model.context_offset.copy_(torch.as_tensor(offset.astype(np.float32), device=device))
    with torch.no_grad():
        modulation = model.modulation(torch.as_tensor(contexts, device=device)).cpu().numpy()
    return {
        "maximum_absolute_train_mean": float(np.max(np.abs(np.mean(modulation, axis=0)))),
        "maximum_absolute_modulation": float(np.max(np.abs(modulation))),
        "mean_modulation_norm": float(np.mean(np.linalg.norm(modulation, axis=1))),
    }


def repair_context_offsets(project_root, output_root=None, scratch_root=SCRATCH_ROOT, device_name="cpu"):
    """Re-solve frozen zero-mean offsets without changing trained parameters."""
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    device = _device(device_name)
    train_cache = load_cache(cache_path(scratch_root, "train"))
    action_truth = action_distance_matrix(project_root, output_root)
    manifest_path = os.path.join(output_root, "MODEL_TRAINING_MANIFEST.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    repaired = []
    for entry in manifest["entries"]:
        metadata = entry["metadata"]
        if metadata.get("method") != CONTEXT_METHOD:
            continue
        path = os.path.join(output_root, entry["path"])
        model, loaded_metadata = load_context_checkpoint(path, device)
        control = str(metadata["control"])
        seed = int(metadata["seed"])
        arrays = controlled_training_arrays(train_cache, control, seed, action_truth)
        audit = solve_zero_mean_offset(model, arrays["context"], device)
        loaded_metadata["zero_mean_offset_audit"] = audit
        loaded_metadata["offset_solver"] = "per-dimension observed-logit bracket with 20-unit exterior margin"
        saved = _save_checkpoint(path, model, loaded_metadata)
        entry["sha256"] = saved["sha256"]
        entry["metadata"] = loaded_metadata
        repaired.append({"path": entry["path"], "sha256": saved["sha256"], "audit": audit})
    manifest["entries"] = manifest["entries"]
    manifest["context_offset_repair_before_development"] = {
        "reason": "fixed [-40,40] root bracket did not enclose six trained-logit roots",
        "trained_parameters_changed": False,
        "optimizer_steps_changed": False,
        "repaired_checkpoints": repaired,
    }
    atomic_json(manifest_path, manifest)
    return manifest["context_offset_repair_before_development"]


def precompute_frozen_embeddings(model, arrays, cache, device, batch_states=16):
    """Cache frozen B1/B2 action embeddings for fast modulator-only training."""
    torch = _torch()
    targets_np = np.asarray(cache["target_residual"], dtype=np.float32)
    candidates_np = np.asarray(cache["candidate_residual"], dtype=np.float32)
    target_rows = []
    candidate_rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(arrays["nominal"]), int(batch_states)):
            stop = min(start + int(batch_states), len(arrays["nominal"]))
            count = stop - start
            nominal = torch.as_tensor(arrays["nominal"][start:stop], device=device)
            target_nominal = nominal[:, None, :].expand(-1, len(targets_np), -1)
            candidate_nominal = nominal[:, None, :].expand(-1, len(candidates_np), -1)
            targets = torch.as_tensor(
                np.broadcast_to(targets_np[None], (count,) + targets_np.shape).copy(),
                device=device,
            )
            candidates = torch.as_tensor(
                np.broadcast_to(candidates_np[None], (count,) + candidates_np.shape).copy(),
                device=device,
            )
            target_rows.append(model.base.encode(target_nominal, targets).cpu().numpy())
            candidate_rows.append(model.base.encode(candidate_nominal, candidates).cpu().numpy())
    arrays["target_embedding"] = np.concatenate(target_rows, axis=0).astype(np.float32)
    arrays["candidate_embedding"] = np.concatenate(candidate_rows, axis=0).astype(np.float32)
    return arrays


def _save_checkpoint(path, model, metadata):
    torch = _torch()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    return {
        "path": path,
        "sha256": sha256_file(path),
        "metadata": metadata,
    }


def load_static_checkpoint(path, device):
    torch = _torch()
    payload = torch.load(path, map_location=device, weights_only=False)
    model = create_static_metric().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["metadata"]


def load_context_checkpoint(path, device):
    torch = _torch()
    payload = torch.load(path, map_location=device, weights_only=False)
    base = create_static_metric()
    model = create_context_metric(base).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["metadata"]


def predict_static(model, cache, device, batch_states=8):
    torch = _torch()
    result = []
    targets_np = np.asarray(cache["target_residual"], dtype=np.float32)
    candidates_np = np.asarray(cache["candidate_residual"], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(cache["context"]), batch_states):
            stop = min(start + batch_states, len(cache["context"]))
            count = stop - start
            nominal = torch.as_tensor(cache["nominal_action"][start:stop], device=device)
            targets = torch.as_tensor(
                np.broadcast_to(targets_np[None, :, :], (count,) + targets_np.shape).copy(),
                device=device,
            )
            candidates = torch.as_tensor(
                np.broadcast_to(candidates_np[None, :, :], (count,) + candidates_np.shape).copy(),
                device=device,
            )
            result.append(model.matrix_distance(nominal, targets, candidates).cpu().numpy())
    return np.concatenate(result, axis=0)


def predict_context(model, cache, device, context_override=None, nominal_override=None, batch_states=8):
    torch = _torch()
    result = []
    contexts = np.asarray(context_override if context_override is not None else cache["context"], dtype=np.float32)
    nominals = np.asarray(nominal_override if nominal_override is not None else cache["nominal_action"], dtype=np.float32)
    targets_np = np.asarray(cache["target_residual"], dtype=np.float32)
    candidates_np = np.asarray(cache["candidate_residual"], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(contexts), batch_states):
            stop = min(start + batch_states, len(contexts))
            count = stop - start
            context = torch.as_tensor(contexts[start:stop], device=device)
            nominal = torch.as_tensor(nominals[start:stop], device=device)
            targets = torch.as_tensor(np.broadcast_to(targets_np[None], (count,) + targets_np.shape).copy(), device=device)
            candidates = torch.as_tensor(np.broadcast_to(candidates_np[None], (count,) + candidates_np.shape).copy(), device=device)
            result.append(model.matrix_distance(context, nominal, targets, candidates).cpu().numpy())
    return np.concatenate(result, axis=0)


def _calibration_metrics(scores, cache):
    truth = np.asarray(cache["true_distance"], dtype=np.float64)
    rows = []
    for state in range(len(truth)):
        for target in range(truth.shape[1]):
            selected = int(np.argmin(scores[state, target]))
            row = ranking_metrics(truth[state, target], scores[state, target], selected)
            row["realized_effect_error"] = float(truth[state, target, selected])
            rows.append(row)
    return {
        "mean_realized_effect_error": float(np.mean([row["realized_effect_error"] for row in rows])),
        "mean_oracle_regret": float(np.mean([row["oracle_regret"] for row in rows])),
        "mean_ndcg_at_16": float(np.mean([row["ndcg_at_16"] for row in rows])),
    }


def train_all(project_root, output_root=None, scratch_root=SCRATCH_ROOT, device_name="cpu"):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    device = _device(device_name)
    torch = _torch()
    if device.type == "cpu":
        torch.set_num_threads(min(16, max(1, os.cpu_count() or 1)))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    train_cache = load_cache(cache_path(scratch_root, "train"))
    calibration = load_cache(cache_path(scratch_root, "calibration"))
    pairs = _load_pairs(output_root)
    action_truth = action_distance_matrix(project_root, output_root)
    entries = []
    # B2 temperature candidates are the sole temperature-selection surface.
    temperature_rows = []
    selected_static_by_tau = {}
    for tau in TEMPERATURE_CANDIDATES:
        models = []
        for seed in MODEL_SEEDS:
            _set_determinism(seed, device)
            model = create_static_metric().to(device)
            arrays = controlled_training_arrays(train_cache, PROPOSED, seed, action_truth)
            training = _train(
                model, arrays, train_cache, action_truth, pairs, tau, seed, device, False,
                "B2_STATIC_CONSEQUENCE",
            )
            relative = os.path.join(
                "STATIC_METRIC_CHECKPOINTS",
                "B2_STATIC_CONSEQUENCE",
                "tau_%0.2f_seed_%d.pt" % (tau, seed),
            )
            metadata = {
                "method": "B2_STATIC_CONSEQUENCE",
                "seed": int(seed),
                "temperature": float(tau),
                "training": training,
                "consequence_labels_used": True,
                "future_inputs_used": False,
            }
            saved = _save_checkpoint(os.path.join(output_root, relative), model, metadata)
            saved["path"] = relative
            entries.append(saved)
            models.append(model)
        score = np.mean([predict_static(model, calibration, device) for model in models], axis=0)
        metrics = _calibration_metrics(score, calibration)
        temperature_rows.append({"temperature": float(tau), **metrics})
        selected_static_by_tau[float(tau)] = models
    selected_row = min(
        temperature_rows,
        key=lambda row: (
            row["mean_realized_effect_error"],
            row["mean_oracle_regret"],
            -row["mean_ndcg_at_16"],
            row["temperature"],
        ),
    )
    selected_tau = float(selected_row["temperature"])
    atomic_json(
        os.path.join(output_root, "TEMPERATURE_CALIBRATION.json"),
        {
            "split": "calibration episodes 32-35 only",
            "criterion": [
                "lowest mean realized BALANCED_TASK_EFFECT",
                "lowest mean oracle regret",
                "highest NDCG@16",
                "smallest temperature",
            ],
            "candidates": temperature_rows,
            "selected_temperature": selected_tau,
            "development_read": False,
        },
    )
    # B1 is trained only after the calibration-only temperature is fixed.
    b1_models = []
    action_arrays = controlled_training_arrays(train_cache, "ACTION_ONLY", MODEL_SEEDS[0], action_truth)
    action_arrays["context"] = np.asarray(train_cache["context"], dtype=np.float32)
    action_arrays["nominal"] = np.asarray(train_cache["nominal_action"], dtype=np.float32)
    for seed in MODEL_SEEDS:
        _set_determinism(seed, device)
        model = create_static_metric().to(device)
        training = _train(
            model, action_arrays, train_cache, action_truth, pairs, selected_tau, seed,
            device, False, "B1_ACTION_ONLY",
        )
        relative = os.path.join(
            "STATIC_METRIC_CHECKPOINTS", "B1_ACTION_ONLY", "seed_%d.pt" % seed
        )
        metadata = {
            "method": "B1_ACTION_ONLY",
            "seed": int(seed),
            "temperature": selected_tau,
            "training": training,
            "consequence_labels_used": False,
            "strict_consequence_reversal_tuples_used": False,
            "future_inputs_used": False,
        }
        saved = _save_checkpoint(os.path.join(output_root, relative), model, metadata)
        saved["path"] = relative
        entries.append(saved)
        b1_models.append(model)
    # Train P1 and every matched control with identical total architecture and
    # identical 2,500 optimizer-step budget.  Only modulator weights train.
    b2_models = selected_static_by_tau[selected_tau]
    context_entries = []
    for control in (PROPOSED,) + tuple(MATCHED_CONTROLS):
        for model_index, seed in enumerate(MODEL_SEEDS):
            base_source = b1_models[model_index] if control == "ACTION_ONLY" else b2_models[model_index]
            base = create_static_metric().to(device)
            base.load_state_dict(base_source.state_dict())
            model = create_context_metric(base, train_cache["context"].shape[1]).to(device)
            _set_determinism(seed, device)
            # Reinitialize only the trainable modulator after setting the seed.
            for module in model.modulator.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            arrays = controlled_training_arrays(train_cache, control, seed, action_truth)
            arrays = precompute_frozen_embeddings(model, arrays, train_cache, device)
            training = _train(
                model, arrays, train_cache, action_truth, pairs, selected_tau, seed,
                device, True, "%s__%s" % (CONTEXT_METHOD, control),
            )
            offset_audit = solve_zero_mean_offset(model, arrays["context"], device)
            relative = os.path.join(
                "CONTEXT_METRIC_CHECKPOINTS", control, "seed_%d.pt" % seed
            )
            metadata = {
                "method": CONTEXT_METHOD,
                "control": control,
                "seed": int(seed),
                "temperature": selected_tau,
                "training": training,
                "zero_mean_offset_audit": offset_audit,
                "base_source": "B1_ACTION_ONLY" if control == "ACTION_ONLY" else "B2_STATIC_CONSEQUENCE",
                "base_frozen": True,
                "only_modulator_trained": True,
                "future_inputs_used": False,
                "phase_input_used": control == "PHASE_ONLY",
            }
            saved = _save_checkpoint(os.path.join(output_root, relative), model, metadata)
            saved["path"] = relative
            entries.append(saved)
            context_entries.append(saved)
    parameter_counts = {
        control: sorted(
            {
                entry["metadata"]["training"]["parameter_count"]
                for entry in context_entries
                if entry["metadata"]["control"] == control
            }
        )
        for control in (PROPOSED,) + tuple(MATCHED_CONTROLS)
    }
    if len({tuple(value) for value in parameter_counts.values()}) != 1:
        raise RuntimeError("matched controls changed P1 parameter count")
    manifest = {
        "selected_temperature": selected_tau,
        "temperature_calibration_artifact": "TEMPERATURE_CALIBRATION.json",
        "device": str(device),
        "model_seeds": list(MODEL_SEEDS),
        "training_steps_per_final_checkpoint": TRAINING_STEPS,
        "entries": entries,
        "matched_parameter_counts": parameter_counts,
        "development_data_read": False,
        "pai_jobs_submitted": 0,
        "torch_threads": int(torch.get_num_threads()),
    }
    atomic_json(os.path.join(output_root, "MODEL_TRAINING_MANIFEST.json"), manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--scratch-root", default=SCRATCH_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repair-context-offsets", action="store_true")
    args = parser.parse_args(argv)
    if args.repair_context_offsets:
        result = repair_context_offsets(args.project_root, args.output_root, args.scratch_root, args.device)
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        result = train_all(args.project_root, args.output_root, args.scratch_root, args.device)
        print(json.dumps({"selected_temperature": result["selected_temperature"], "checkpoints": len(result["entries"])}, indent=2))


if __name__ == "__main__":
    main()
