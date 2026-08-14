"""Small state-based predictors for the frozen Stage 3 mechanism audit.

These models predict/rank physical consequences.  They are not policies and
never consume images, future outcomes, or demonstration phase in proposed
methods.
"""

from __future__ import annotations

import copy
import math
import os

import numpy as np

from .stage2_analysis import CONTINUOUS_INDICES, PRIMARY_GROUPS
from .stage3_config import (
    C0_BATCH_SIZE,
    C0_LEARNING_RATE,
    C0_MAX_EPOCHS,
    C0_PATIENCE,
    C0_WEIGHT_DECAY,
    CONTACT_LOSS_WEIGHT,
    EMBEDDING_DIM,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_DELTA,
    PAIR_BATCH_SIZE,
    PATIENCE,
    PREDICTOR_BATCH_SIZE,
    SOFT_EXPERTS,
    WEIGHT_DECAY,
)


def _torch():
    import torch

    return torch


def _nn():
    import torch.nn as nn

    return nn


def _make_mlp(input_dim, hidden, output_dim, final_activation=None):
    nn = _nn()
    layers = []
    previous = int(input_dim)
    for width in hidden:
        layers.extend((nn.Linear(previous, int(width)), nn.GELU()))
        previous = int(width)
    layers.append(nn.Linear(previous, int(output_dim)))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class VectorPredictorBase:
    pass


def create_vector_predictor(context_dim, residual_dim, output_dim, hidden):
    nn = _nn()

    class VectorPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = _make_mlp(
                context_dim + residual_dim,
                hidden,
                output_dim + 4,
            )

        def forward(self, context, residual):
            value = self.network(_torch().cat((context, residual), dim=1))
            return value[:, :output_dim], value[:, output_dim:]

    return VectorPredictor()


def create_temporal_vector_predictor(context_dim, output_dim, temporal_hidden):
    torch = _torch()
    nn = _nn()

    class TemporalVectorPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.context = _make_mlp(context_dim - 24, (128,), 128)
            self.nominal = nn.GRU(6, int(temporal_hidden), batch_first=True)
            self.residual = nn.GRU(6, int(temporal_hidden), batch_first=True)
            self.output = _make_mlp(
                128 + 2 * int(temporal_hidden),
                (int(temporal_hidden),),
                output_dim + 4,
            )

        def forward(self, context, residual):
            # Nominal action occupies the frozen [293:317] context slice.
            nominal = context[:, 293:317].reshape(-1, 4, 6)
            context_without_nominal = torch.cat((context[:, :293], context[:, 317:]), dim=1)
            context_latent = self.context(context_without_nominal)
            _, nominal_hidden = self.nominal(nominal)
            _, residual_hidden = self.residual(residual.reshape(-1, 4, 6))
            value = self.output(
                torch.cat(
                    (context_latent, nominal_hidden[-1], residual_hidden[-1]), dim=1
                )
            )
            return value[:, :output_dim], value[:, output_dim:]

    return TemporalVectorPredictor()


def create_c0_predictor(input_dim, output_dim, hidden):
    nn = _nn()

    class C0Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = _make_mlp(input_dim, hidden, output_dim + 4)

        def forward(self, value):
            output = self.network(value)
            return output[:, :output_dim], output[:, output_dim:]

    return C0Predictor()


def create_biencoder(context_dim, residual_dim=24, embedding_dim=EMBEDDING_DIM):
    torch = _torch()
    nn = _nn()

    class BiEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.context_network = _make_mlp(context_dim, (192, 128), 96)
            self.action_network = _make_mlp(
                96 + residual_dim, (128, 96), int(embedding_dim)
            )

        def embed(self, context, residual):
            latent = self.context_network(context)
            return self.action_network(torch.cat((latent, residual), dim=1))

        def forward(self, context, target, candidate):
            target_embedding = self.embed(context, target)
            candidate_embedding = self.embed(context, candidate)
            return torch.linalg.vector_norm(
                target_embedding - candidate_embedding, dim=1
            )

    return BiEncoder()


def _symmetric_pair_features(context, target, candidate):
    torch = _torch()
    mean = 0.5 * (target + candidate)
    difference = torch.abs(target - candidate)
    product = target * candidate
    return torch.cat((context, mean, difference, product), dim=1)


def create_pair_ranker(context_dim, hidden=(192, 192)):
    torch = _torch()
    nn = _nn()

    class PairRanker(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = _make_mlp(context_dim + 72, hidden, 1)

        def forward(self, context, target, candidate):
            feature = _symmetric_pair_features(context, target, candidate)
            scale = torch.nn.functional.softplus(self.network(feature).squeeze(1))
            # Exact self-distance is architectural rather than approximate.
            return torch.linalg.vector_norm(target - candidate, dim=1) * scale

    return PairRanker()


def create_soft_mixture_ranker(context_dim, hidden=(192, 192), experts=SOFT_EXPERTS):
    torch = _torch()
    nn = _nn()

    class SoftMixtureRanker(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = _make_mlp(context_dim, (64,), int(experts))
            self.trunk = _make_mlp(context_dim + 72, hidden, int(hidden[-1]))
            self.heads = nn.ModuleList(
                [nn.Linear(int(hidden[-1]), 1) for _ in range(int(experts))]
            )

        def forward(self, context, target, candidate, return_router=False):
            router_logits = self.router(context)
            weights = torch.softmax(router_logits, dim=1)
            latent = self.trunk(_symmetric_pair_features(context, target, candidate))
            expert_scale = torch.cat(
                [torch.nn.functional.softplus(head(latent)) for head in self.heads], dim=1
            )
            scale = torch.sum(weights * expert_scale, dim=1)
            value = torch.linalg.vector_norm(target - candidate, dim=1) * scale
            if return_router:
                return value, router_logits, weights
            return value

    return SoftMixtureRanker()


def create_action_autoencoder(context_dim, residual_dim=24, latent_dim=32):
    torch = _torch()
    nn = _nn()

    class ActionAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _make_mlp(context_dim + residual_dim, (128, 128), latent_dim)
            self.decoder = _make_mlp(context_dim + latent_dim, (128, 128), residual_dim)

        def encode(self, context, residual):
            return self.encoder(torch.cat((context, residual), dim=1))

        def forward(self, context, residual):
            latent = self.encode(context, residual)
            decoded = self.decoder(torch.cat((context, latent), dim=1))
            return decoded, latent

    return ActionAutoencoder()


def _set_determinism(seed, device):
    torch = _torch()
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _clone_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _group_columns():
    index_to_column = {int(index): column for column, index in enumerate(CONTINUOUS_INDICES)}
    return [
        [index_to_column[int(index)] for index in indices]
        for indices in PRIMARY_GROUPS.values()
    ]


GROUP_COLUMNS = _group_columns()


def balanced_vector_loss(prediction, target, target_mask):
    torch = _torch()
    losses = []
    for columns in GROUP_COLUMNS:
        active = target_mask[:, columns].float()
        element = torch.nn.functional.smooth_l1_loss(
            prediction[:, columns], target[:, columns], reduction="none"
        )
        loss = torch.sum(element * active, dim=1) / torch.clamp(
            torch.sum(active, dim=1), min=1.0
        )
        losses.append(loss)
    return torch.mean(torch.stack(losses, dim=1))


def _vector_epoch(model, dataset, device, optimizer=None, batch_size=PREDICTOR_BATCH_SIZE, seed=0):
    torch = _torch()
    training = optimizer is not None
    model.train(training)
    order = np.arange(len(dataset["state_index"]), dtype=np.int64)
    if training:
        order = np.random.RandomState(int(seed)).permutation(order)
    losses = []
    for start in range(0, len(order), int(batch_size)):
        index = order[start : start + int(batch_size)]
        state_ids = dataset["state_index"][index]
        context = torch.as_tensor(dataset["contexts"][state_ids], device=device)
        residual = torch.as_tensor(dataset["residual"][index] / 0.12, device=device)
        target = torch.as_tensor(dataset["target"][index], device=device)
        target_mask = torch.as_tensor(dataset["target_mask"][index], device=device)
        contact = torch.as_tensor(dataset["contact"][index], device=device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction, logits = model(context, residual)
        loss = balanced_vector_loss(prediction, target, target_mask)
        loss = loss + CONTACT_LOSS_WEIGHT * torch.nn.functional.cross_entropy(logits, contact)
        if training:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train_vector_model(
    train_dataset,
    calibration_dataset,
    family,
    hidden,
    seed,
    device,
    max_epochs=MAX_EPOCHS,
    patience_limit=PATIENCE,
):
    _set_determinism(seed, device)
    output_dim = train_dataset["target"].shape[1]
    if family == "C1_NC_VECTOR":
        model = create_vector_predictor(
            train_dataset["contexts"].shape[1], 24, output_dim, hidden
        )
    elif family == "C2_NC_TEMPORAL_VECTOR":
        model = create_temporal_vector_predictor(
            train_dataset["contexts"].shape[1], output_dim, int(hidden)
        )
    else:
        raise KeyError(family)
    model = model.to(device)
    optimizer = _torch().optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    patience = 0
    trace = []
    for epoch in range(int(max_epochs)):
        train_loss = _vector_epoch(
            model, train_dataset, device, optimizer, seed=seed + epoch
        )
        with _torch().no_grad():
            calibration_loss = _vector_epoch(model, calibration_dataset, device)
        trace.append(
            {
                "epoch": int(epoch),
                "train_loss": train_loss,
                "calibration_loss": calibration_loss,
            }
        )
        if calibration_loss < best_loss - MIN_DELTA:
            best_loss = calibration_loss
            best_state = _clone_state(model)
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= int(patience_limit):
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "family": family,
        "hidden": list(hidden) if isinstance(hidden, (tuple, list)) else int(hidden),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_calibration_loss": float(best_loss),
        "epochs_ran": len(trace),
        "trace": trace,
    }


def predict_vector_model(model, contexts, state_index, residual, device, batch_size=8192):
    torch = _torch()
    predictions = []
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(residual), int(batch_size)):
            stop = min(start + int(batch_size), len(residual))
            state_ids = state_index[start:stop]
            context = torch.as_tensor(contexts[state_ids], device=device)
            action = torch.as_tensor(residual[start:stop] / 0.12, device=device)
            prediction, logits = model(context, action)
            predictions.append(prediction.cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    probability = np.concatenate(probabilities)
    return {
        "effect": np.concatenate(predictions),
        "probability": probability,
        "mode": np.argmax(probability, axis=1),
    }


def _pair_objective(scores, true_distance, mask, objective):
    torch = _torch()
    active = mask.float()
    distance = torch.nn.functional.smooth_l1_loss(
        scores, true_distance, reduction="none"
    )
    distance_loss = torch.sum(distance * active) / torch.clamp(torch.sum(active), min=1.0)

    positives = scores[:, :8]
    negative_mask = mask[:, 8:]
    negatives = scores[:, 8:]
    comparison = torch.nn.functional.softplus(
        positives[:, :, None] - negatives[:, None, :]
    )
    comparison_mask = negative_mask[:, None, :].float()
    pairwise_loss = torch.sum(comparison * comparison_mask) / torch.clamp(
        torch.sum(comparison_mask) * positives.shape[1], min=1.0
    )

    true_logits = -true_distance / float(objective["tau_true"])
    model_logits = -scores / float(objective["tau_model"])
    true_logits = true_logits.masked_fill(~mask, -1e9)
    model_logits = model_logits.masked_fill(~mask, -1e9)
    true_probability = torch.softmax(true_logits, dim=1)
    model_log_probability = torch.log_softmax(model_logits, dim=1)
    listwise_loss = -torch.mean(torch.sum(true_probability * model_log_probability, dim=1))
    total = (
        float(objective["lambda_distance"]) * distance_loss
        + float(objective["lambda_pairwise"]) * pairwise_loss
        + float(objective["lambda_listwise"]) * listwise_loss
    )
    return total, distance_loss, pairwise_loss, listwise_loss


def _pair_epoch(
    model,
    dataset,
    objective,
    device,
    optimizer=None,
    seed=0,
    route_labels=None,
):
    torch = _torch()
    training = optimizer is not None
    model.train(training)
    order = np.arange(len(dataset["state_index"]), dtype=np.int64)
    if training:
        order = np.random.RandomState(int(seed)).permutation(order)
    candidates_per_group = dataset["candidate_residual"].shape[1]
    group_batch = max(1, int(PAIR_BATCH_SIZE) // int(candidates_per_group))
    losses = []
    parts = []
    for start in range(0, len(order), group_batch):
        group_ids = order[start : start + group_batch]
        state_ids = dataset["state_index"][group_ids]
        count = len(group_ids)
        context_group = torch.as_tensor(dataset["contexts"][state_ids], device=device)
        context = context_group[:, None, :].expand(-1, candidates_per_group, -1)
        context = context.reshape(count * candidates_per_group, -1)
        target_group = torch.as_tensor(dataset["target_residual"][group_ids] / 0.12, device=device)
        target = target_group[:, None, :].expand(-1, candidates_per_group, -1)
        target = target.reshape(count * candidates_per_group, -1)
        candidate = torch.as_tensor(
            dataset["candidate_residual"][group_ids] / 0.12, device=device
        ).reshape(count * candidates_per_group, -1)
        true_distance = torch.as_tensor(dataset["true_distance"][group_ids], device=device)
        mask = torch.as_tensor(dataset["candidate_mask"][group_ids], device=device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        if route_labels is not None:
            flat_score, router_logits, _ = model(
                context, target, candidate, return_router=True
            )
            router_group = router_logits.reshape(count, candidates_per_group, -1)[:, 0]
        else:
            flat_score = model(context, target, candidate)
            router_group = None
        score = flat_score.reshape(count, candidates_per_group)
        total, distance_loss, pairwise_loss, listwise_loss = _pair_objective(
            score, true_distance, mask, objective
        )
        if route_labels is not None:
            label = torch.as_tensor(route_labels[state_ids], device=device)
            route_loss = torch.nn.functional.cross_entropy(router_group, label)
            total = total + 0.10 * route_loss
        else:
            route_loss = torch.zeros((), device=device)
        if training:
            total.backward()
            optimizer.step()
        losses.append(float(total.detach().cpu()))
        parts.append(
            (
                float(distance_loss.detach().cpu()),
                float(pairwise_loss.detach().cpu()),
                float(listwise_loss.detach().cpu()),
                float(route_loss.detach().cpu()),
            )
        )
    mean_parts = np.mean(np.asarray(parts), axis=0)
    return float(np.mean(losses)), {
        "distance": float(mean_parts[0]),
        "pairwise": float(mean_parts[1]),
        "listwise": float(mean_parts[2]),
        "routing": float(mean_parts[3]),
    }


def train_pair_model(
    train_dataset,
    calibration_dataset,
    family,
    objective,
    seed,
    device,
    route_labels_train=None,
    route_labels_calibration=None,
):
    _set_determinism(seed, device)
    context_dim = train_dataset["contexts"].shape[1]
    if family == "C3_NC_BIENCODER":
        model = create_biencoder(context_dim)
    elif family in (
        "C4_NC_PAIR_RANKER",
        "no_nominal_action",
        "nominal_action_shuffled_within_task",
        "state_shuffled_within_task",
        "joint_state_nominal_shuffled_within_task",
        "history_shuffled",
        "consequence_labels_shuffled",
        "action_only_pair_ranker",
    ):
        model = create_pair_ranker(context_dim)
    elif family in ("C6_SOFT_MIXTURE_NCER_AA", "soft_routing_labels_shuffled"):
        model = create_soft_mixture_ranker(context_dim)
    else:
        raise KeyError(family)
    model = model.to(device)
    optimizer = _torch().optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    patience = 0
    trace = []
    for epoch in range(MAX_EPOCHS):
        train_loss, train_parts = _pair_epoch(
            model,
            train_dataset,
            objective,
            device,
            optimizer=optimizer,
            seed=seed + epoch,
            route_labels=route_labels_train,
        )
        with _torch().no_grad():
            calibration_loss, calibration_parts = _pair_epoch(
                model,
                calibration_dataset,
                objective,
                device,
                route_labels=route_labels_calibration,
            )
        trace.append(
            {
                "epoch": int(epoch),
                "train_loss": train_loss,
                "calibration_loss": calibration_loss,
                "train_parts": train_parts,
                "calibration_parts": calibration_parts,
            }
        )
        if calibration_loss < best_loss - MIN_DELTA:
            best_loss = calibration_loss
            best_state = _clone_state(model)
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "family": family,
        "objective": dict(objective),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_calibration_loss": float(best_loss),
        "epochs_ran": len(trace),
        "trace": trace,
    }


def score_pairs(model, context, target, candidates, device, batch_size=8192):
    torch = _torch()
    target = np.asarray(target, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    context = np.asarray(context, dtype=np.float32)
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(candidates), int(batch_size)):
            stop = min(start + int(batch_size), len(candidates))
            ctx = torch.as_tensor(
                np.repeat(context[None, :], stop - start, axis=0), device=device
            )
            tgt = torch.as_tensor(
                np.repeat((target / 0.12)[None, :], stop - start, axis=0), device=device
            )
            cand = torch.as_tensor(candidates[start:stop] / 0.12, device=device)
            output.append(model(ctx, tgt, cand).cpu().numpy())
    return np.concatenate(output)


def embed_actions(model, context, residuals, device, batch_size=8192):
    torch = _torch()
    residuals = np.asarray(residuals, dtype=np.float32)
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(residuals), int(batch_size)):
            stop = min(start + int(batch_size), len(residuals))
            ctx = torch.as_tensor(
                np.repeat(context[None, :], stop - start, axis=0), device=device
            )
            action = torch.as_tensor(residuals[start:stop] / 0.12, device=device)
            output.append(model.embed(ctx, action).cpu().numpy())
    return np.concatenate(output)


def train_action_autoencoder(train_contexts, calibration_contexts, action_bank, seed, device):
    torch = _torch()
    _set_determinism(seed, device)
    model = create_action_autoencoder(train_contexts.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    train_state = np.repeat(np.arange(len(train_contexts)), len(action_bank))
    train_action = np.tile(action_bank.astype(np.float32) / 0.12, (len(train_contexts), 1))
    cal_state = np.repeat(np.arange(len(calibration_contexts)), len(action_bank))
    cal_action = np.tile(action_bank.astype(np.float32) / 0.12, (len(calibration_contexts), 1))
    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    patience = 0
    trace = []
    for epoch in range(MAX_EPOCHS):
        order = np.random.RandomState(seed + epoch).permutation(len(train_state))
        model.train()
        losses = []
        for start in range(0, len(order), PREDICTOR_BATCH_SIZE):
            index = order[start : start + PREDICTOR_BATCH_SIZE]
            context = torch.as_tensor(train_contexts[train_state[index]], device=device)
            action = torch.as_tensor(train_action[index], device=device)
            decoded, _ = model(context, action)
            loss = torch.nn.functional.mse_loss(decoded, action)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        cal_losses = []
        with torch.no_grad():
            for start in range(0, len(cal_state), PREDICTOR_BATCH_SIZE):
                stop = min(start + PREDICTOR_BATCH_SIZE, len(cal_state))
                context = torch.as_tensor(
                    calibration_contexts[cal_state[start:stop]], device=device
                )
                action = torch.as_tensor(cal_action[start:stop], device=device)
                decoded, _ = model(context, action)
                cal_losses.append(float(torch.nn.functional.mse_loss(decoded, action).cpu()))
        calibration_loss = float(np.mean(cal_losses))
        trace.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)),
                "calibration_loss": calibration_loss,
            }
        )
        if calibration_loss < best_loss - MIN_DELTA:
            best_loss = calibration_loss
            best_state = _clone_state(model)
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "family": "B4_state_action_vq",
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_calibration_loss": float(best_loss),
        "epochs_ran": len(trace),
        "trace": trace,
    }


def encode_action_autoencoder(model, context, residuals, device, batch_size=8192):
    torch = _torch()
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(residuals), int(batch_size)):
            stop = min(start + int(batch_size), len(residuals))
            ctx = torch.as_tensor(
                np.repeat(context[None, :], stop - start, axis=0), device=device
            )
            action = torch.as_tensor(residuals[start:stop] / 0.12, device=device)
            output.append(model.encode(ctx, action).cpu().numpy())
    return np.concatenate(output)


def save_model(path, model, metadata):
    torch = _torch()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
