from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Sequence
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl.aggregators.base import AggregationContext
from fl.scorers.base import ScoringContext
from fl.datasets.synthetic import ClientDataset
from fl.models import SynthModelSpec, build_synth_model, resolve_synth_model_spec

@dataclass
class RunOutputs:
    metrics_round: pd.DataFrame
    summary: Dict[str, Any]
    client_scores_round: pd.DataFrame
    fedprox_debug_round: pd.DataFrame | None = None
    fedprox_debug: pd.DataFrame | None = None

def run_stub_engine(
    rounds: int,
    n_clients: int,
    dim: int,
    aggregator,
    scorer,
) -> RunOutputs:
    rows = []
    score_rows = []
    start = time.time()
    global_vec = np.zeros((dim,), dtype=np.float64)

    for r in range(rounds):
        # Fake client updates: normal noise around global vector direction
        client_updates = np.random.randn(n_clients, dim).astype(np.float64) * 0.1 + global_vec[None, :] * 0.0

        sctx = ScoringContext(round_idx=r, meta={})
        scorer.fit_round(client_updates, sctx)
        raw_scores = np.asarray(scorer.score_clients(client_updates, sctx), dtype=np.float64).reshape(-1)
        if raw_scores.shape[0] != n_clients:
            raise AssertionError(f"Scorer returned shape {raw_scores.shape}, expected ({n_clients},).")
        semantics = str(sctx.meta.get("score_semantics", getattr(scorer, "score_semantics", "reliability"))).lower()
        harm_scores, reliability = _resolve_harm_and_reliability_from_meta(raw_scores, semantics, sctx.meta)
        if not np.all(np.isfinite(reliability)):
            raise AssertionError("Non-finite reliability in stub engine.")
        scorer_weights = _extract_scorer_weights(sctx.meta, n_clients)
        agg_weights = scorer_weights if scorer_weights is not None else reliability

        actx = AggregationContext(round_idx=r, meta={"score_semantics": semantics})
        update = aggregator.aggregate(client_updates, weights=agg_weights, ctx=actx)
        weight_i_used = _resolve_weight_i_used(actx.meta, n_clients, fallback_weights=agg_weights)
        cos_i = _cosine_to_aggregate_update(client_updates, update)
        if not np.all(np.isfinite(cos_i)):
            raise AssertionError("Non-finite cos_i in stub engine.")
        if not np.all(np.isfinite(weight_i_used)):
            raise AssertionError("Non-finite weight_i in stub engine.")
        if not np.all(np.isfinite(reliability)):
            raise AssertionError("Non-finite reliability in stub engine.")
        if bool(actx.meta.get("weights_sum_to_one", True)):
            if not np.isclose(float(np.sum(weight_i_used)), 1.0, atol=1e-6):
                raise AssertionError("weight_i does not sum to 1 in stub engine.")
        global_vec = global_vec + update

        # Stub metric: negative norm (just to see change)
        metric = -float(np.linalg.norm(global_vec))
        rows.append({"round": r, "stub_metric": metric})
        for i, sc in enumerate(raw_scores):
            score_rows.append(
                {
                    "round": r,
                    "client_id": i,
                    "score": float(sc),
                    "harm_score": float(harm_scores[i]),
                    "reliability": float(reliability[i]),
                    "global_val_loss_before": float("nan"),
                    "global_val_loss_after": float("nan"),
                    "global_val_delta_loss": float("nan"),
                    "cos_i": float(cos_i[i]),
                    "weight_i": float(weight_i_used[i]),
                }
            )

    elapsed = time.time() - start
    metrics_round = pd.DataFrame(rows)
    client_scores_round = pd.DataFrame(score_rows)
    summary = {
        "rounds": rounds,
        "n_clients": n_clients,
        "dim": dim,
        "final_stub_metric": float(metrics_round["stub_metric"].iloc[-1]),
        "runtime_sec": float(elapsed),
    }
    return RunOutputs(metrics_round=metrics_round, summary=summary, client_scores_round=client_scores_round)


def _resolve_device(device_name: str) -> torch.device:
    requested = str(device_name).lower()
    if requested in {"gpu", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _extract_scorer_weights(meta: Dict[str, Any], n: int) -> np.ndarray | None:
    arr = meta.get("scorer_weights")
    if arr is None:
        return None
    weights = np.asarray(arr, dtype=np.float64).reshape(-1)
    if weights.shape[0] != n:
        return None
    weights = np.clip(weights, 0.0, None)
    s = float(weights.sum())
    if s <= 1e-12:
        return np.full((n,), 1.0 / n, dtype=np.float64)
    return weights / s


def _normalize_nonnegative_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    n = int(w.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 1e-12:
        return np.full((n,), 1.0 / n, dtype=np.float64)
    return w / s


def _normalize_to_unit_interval(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) <= eps:
        return np.zeros_like(arr, dtype=np.float64)
    out = (arr - vmin) / (vmax - vmin)
    return np.clip(out, 0.0, 1.0)


def _derive_harm_and_reliability(
    raw_scores: np.ndarray,
    semantics: str,
) -> tuple[np.ndarray, np.ndarray]:
    sem = str(semantics).strip().lower()
    scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    if sem == "harm":
        harm = _normalize_to_unit_interval(scores)
        reliability = 1.0 - harm
        return harm, np.clip(reliability, 0.0, 1.0)
    if sem == "reliability":
        reliability = np.clip(scores, 0.0, 1.0).astype(np.float64, copy=False)
        harm = 1.0 - reliability
        return harm, reliability
    raise ValueError(f"Unknown scorer semantics '{sem}'. Expected 'reliability' or 'harm'.")


def _extract_optional_meta_vector(meta: Dict[str, Any], key: str, n: int) -> np.ndarray | None:
    if key not in meta or meta.get(key) is None:
        return None
    arr = np.asarray(meta[key], dtype=np.float64).reshape(-1)
    if arr.shape[0] != n:
        raise AssertionError(f"ctx.meta['{key}'] has shape {arr.shape}, expected ({n},).")
    return arr


def _extract_optional_meta_vector_or_nan(meta: Dict[str, Any], key: str, n: int) -> np.ndarray:
    arr = _extract_optional_meta_vector(meta, key, n)
    if arr is None:
        return np.full((n,), np.nan, dtype=np.float64)
    return arr


def _extract_optional_meta_scalar(meta: Dict[str, Any], key: str) -> float:
    if key not in meta:
        return float("nan")
    arr = np.asarray(meta[key], dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(arr[0])


def _resolve_harm_and_reliability_from_meta(
    raw_scores: np.ndarray,
    semantics: str,
    meta: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    n = int(scores.shape[0])
    harm, reliability = _derive_harm_and_reliability(scores, semantics)

    harm_override = _extract_optional_meta_vector(meta, "harm_score_i", n)
    if harm_override is not None:
        harm = harm_override

    reliability_override = _extract_optional_meta_vector(meta, "reliability_i", n)
    if reliability_override is not None:
        reliability = np.clip(reliability_override, 0.0, 1.0)

    if not np.all(np.isfinite(harm)):
        raise AssertionError("Found non-finite harm scores.")
    if not np.all(np.isfinite(reliability)):
        raise AssertionError("Found non-finite reliability values.")
    return harm, reliability


def _resolve_weight_i_used(meta: Dict[str, Any], n: int, fallback_weights: np.ndarray) -> np.ndarray:
    arr = meta.get("weight_i_used")
    if arr is None:
        return _normalize_nonnegative_weights(fallback_weights)
    w = np.asarray(arr, dtype=np.float64).reshape(-1)
    if w.shape[0] != n:
        raise AssertionError(f"Aggregator produced weight_i_used shape {w.shape}, expected ({n},).")
    if not np.all(np.isfinite(w)):
        raise AssertionError("Aggregator produced non-finite weight_i_used.")
    return w


def _cosine_to_aggregate_update(client_updates: np.ndarray, aggregate_update: np.ndarray) -> np.ndarray:
    updates = np.asarray(client_updates, dtype=np.float64)
    agg = np.asarray(aggregate_update, dtype=np.float64).reshape(-1)
    if updates.ndim != 2:
        raise ValueError(f"Expected updates shape [n_clients, d], got {updates.shape}")
    n_clients = int(updates.shape[0])
    if n_clients == 0:
        return np.zeros((0,), dtype=np.float64)
    agg_norm = float(np.linalg.norm(agg))
    client_norms = np.linalg.norm(updates, axis=1)
    if agg_norm <= 1e-12:
        return np.zeros((n_clients,), dtype=np.float64)
    numer = updates @ agg
    denom = client_norms * agg_norm
    cos = np.divide(numer, denom + 1e-12)
    cos = np.where(client_norms <= 1e-12, 0.0, cos)
    return np.clip(cos, -1.0, 1.0).astype(np.float64, copy=False)


def _label_hist_binary(y: np.ndarray) -> np.ndarray:
    y_int = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y_int, minlength=2).astype(np.float64)
    return counts / (counts.sum() + 1e-12)


def _client_role(client: ClientDataset) -> str:
    if bool(client.is_attacker):
        return "attacker"
    if bool(client.is_noisy):
        return "noisy"
    return "clean"


def _get_model_params(model: torch.nn.Module) -> np.ndarray:
    chunks = [p.detach().cpu().reshape(-1).numpy() for p in model.parameters()]
    if not chunks:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(chunks, axis=0).astype(np.float64, copy=False)


def _set_model_params(model: torch.nn.Module, params: np.ndarray, device: torch.device) -> None:
    flat = np.asarray(params, dtype=np.float64).reshape(-1)
    expected = int(sum(p.numel() for p in model.parameters()))
    if flat.shape[0] != expected:
        raise ValueError(f"Expected parameter vector of shape ({expected},), got {flat.shape}.")
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = int(p.numel())
            chunk = flat[offset : offset + n]
            tensor = torch.tensor(chunk, dtype=p.dtype, device=device).reshape_as(p)
            p.copy_(tensor)
            offset += n


def _init_global_params(
    model_spec: SynthModelSpec,
    device: torch.device,
) -> np.ndarray:
    if model_spec.name == "linear_logreg_2d":
        return np.zeros((int(model_spec.input_dim) + 1,), dtype=np.float64)
    model = build_synth_model(model_spec).to(device)
    return _get_model_params(model)


def _predict_binary_proba(
    x_data: np.ndarray,
    params: np.ndarray,
    model_spec: SynthModelSpec,
    device: torch.device,
) -> np.ndarray:
    model = build_synth_model(model_spec).to(device)
    _set_model_params(model, params, device)
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x_data, dtype=torch.float32, device=device)
        logits = model(x)
        if model_spec.output_dim == 1:
            p1 = torch.sigmoid(logits).reshape(-1)
            probs = torch.stack([1.0 - p1, p1], dim=1).cpu().numpy().astype(np.float64, copy=False)
        elif model_spec.output_dim == 2:
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError("Expected logits shape [n,2] for model with output_dim=2.")
            probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64, copy=False)
        else:
            raise ValueError(f"Unsupported output_dim={model_spec.output_dim} for binary classification.")
    return np.clip(probs, 1e-12, 1.0)


def _build_optimizer(name: str, model: torch.nn.Module, lr: float):
    opt_name = str(name).lower()
    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    raise ValueError(f"Unsupported optimizer: {name}")


def _grad_norm_total(model: torch.nn.Module) -> float:
    sq_sum = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().to(dtype=torch.float64)
        sq_sum += float(torch.sum(g * g).item())
    if sq_sum <= 0.0:
        return 0.0
    return float(np.sqrt(sq_sum))


def _delta_l2_to_ref(model: torch.nn.Module, p0_params: list[torch.Tensor]) -> torch.Tensor:
    # Use float64 accumulation for stability on tiny/large deltas.
    delta = torch.zeros((), dtype=torch.float64, device=next(model.parameters()).device)
    for p, p0 in zip(model.parameters(), p0_params):
        diff = p.to(dtype=torch.float64) - p0
        delta = delta + torch.sum(diff * diff)
    return delta


def _train_local_update(
    client: ClientDataset,
    global_params: np.ndarray,
    model_spec: SynthModelSpec,
    local_epochs: int,
    batch_size: int,
    lr: float,
    fl_method: str,
    fedprox_mu: float,
    optimizer_name: str,
    device: torch.device,
    track_debug: bool = False,
    track_step_debug: bool = False,
    round_idx: int = -1,
) -> tuple[
    np.ndarray,
    float,
    float,
    float,
    float,
    float,
    int,
    float,
    float,
    int,
    int,
    Dict[str, float] | None,
    list[Dict[str, float | int]],
]:
    method = str(fl_method).strip().lower()
    if method not in {"fedavg", "fedprox"}:
        raise ValueError(f"Unsupported fl_method: {fl_method}. Expected one of: fedavg, fedprox")
    mu = float(fedprox_mu)
    if mu < 0.0:
        raise ValueError("fedprox_mu must be >= 0.0")

    model = build_synth_model(model_spec).to(device)
    _set_model_params(model, global_params, device)
    model.train()
    # Snapshot at start of round; do not keep live parameter references.
    p0_params = [p.detach().clone().to(dtype=torch.float64) for p in model.parameters()]

    x_train = torch.tensor(client.x_train, dtype=torch.float32, device=device)
    if model_spec.output_dim == 1:
        y_train = torch.tensor(client.y_train, dtype=torch.float32, device=device).reshape(-1, 1)
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss()
    elif model_spec.output_dim == 2:
        y_train = torch.tensor(client.y_train, dtype=torch.long, device=device).reshape(-1)
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported output_dim={model_spec.output_dim} for binary classification.")
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

    optimizer = _build_optimizer(optimizer_name, model, lr)
    ce_loss_sum = 0.0
    prox_term_sum = 0.0
    delta_l2_sum = 0.0
    delta_l2_min = float("inf")
    delta_l2_max = float("-inf")
    n_steps = 0
    prox_term_min = float("inf")
    prox_term_max = float("-inf")
    prox_term_nan_count = 0
    prox_term_inf_count = 0
    delta_l2_before_sum = 0.0
    delta_l2_after_sum = 0.0
    prox_before_sum = 0.0
    prox_after_sum = 0.0
    step_debug_rows: list[Dict[str, float | int]] = []
    step_idx = 0

    for _ in range(local_epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            ce_loss = criterion(logits, yb)
            delta_l2_before = _delta_l2_to_ref(model, p0_params)
            delta_l2_before_val = float(delta_l2_before.item())
            delta_l2_sum += delta_l2_before_val
            delta_l2_min = min(delta_l2_min, delta_l2_before_val)
            delta_l2_max = max(delta_l2_max, delta_l2_before_val)

            prox_term = torch.zeros((), dtype=delta_l2_before.dtype, device=device)
            if method == "fedprox" and mu > 0.0:
                # Keep prox term on-graph (no .item()) so it contributes gradient.
                prox_term = 0.5 * mu * delta_l2_before

            # Check non-finite on tensor term; only detach for logging.
            if not bool(torch.isfinite(prox_term).all().item()):
                if bool(torch.isnan(prox_term).any().item()):
                    prox_term_nan_count += 1
                if bool(torch.isinf(prox_term).any().item()):
                    prox_term_inf_count += 1
                prox_term_for_loss = torch.zeros((), dtype=ce_loss.dtype, device=device)
                prox_safe_val = 0.0
            else:
                prox_term_for_loss = prox_term.to(dtype=ce_loss.dtype)
                if not bool(torch.isfinite(prox_term_for_loss).all().item()):
                    if bool(torch.isnan(prox_term_for_loss).any().item()):
                        prox_term_nan_count += 1
                    if bool(torch.isinf(prox_term_for_loss).any().item()):
                        prox_term_inf_count += 1
                    prox_term_for_loss = torch.zeros((), dtype=ce_loss.dtype, device=device)
                    prox_safe_val = 0.0
                else:
                    prox_safe_val = float(prox_term_for_loss.detach().item())

            loss = ce_loss + prox_term_for_loss
            ce_loss_val = float(ce_loss.item())
            total_loss_val = float(loss.item())
            ce_loss_sum += float(ce_loss.item())
            prox_term_sum += prox_safe_val
            n_steps += 1
            prox_term_min = min(prox_term_min, prox_safe_val)
            prox_term_max = max(prox_term_max, prox_safe_val)
            if track_debug:
                delta_l2_before_sum += delta_l2_before_val
                prox_before_sum += prox_safe_val
            loss.backward()
            grad_norm = _grad_norm_total(model) if track_step_debug else 0.0
            optimizer.step()
            if track_step_debug:
                delta_l2_after = _delta_l2_to_ref(model, p0_params)
                delta_l2_after_val = float(delta_l2_after.item())
                param_delta_l2 = float(np.sqrt(max(delta_l2_after_val, 0.0)))
                step_debug_rows.append(
                    {
                        "round": int(round_idx),
                        "client_id": int(client.client_id),
                        "step": int(step_idx),
                        "ce_loss": ce_loss_val,
                        "prox_loss": float(prox_safe_val),
                        "total_loss": total_loss_val,
                        "grad_norm": float(grad_norm),
                        "param_delta_l2": param_delta_l2,
                    }
                )
                step_idx += 1
            if track_debug:
                delta_l2_after = _delta_l2_to_ref(model, p0_params)
                delta_l2_after_sum += float(delta_l2_after.item())
                prox_after_sum += float((0.5 * mu * delta_l2_after).item()) if mu > 0.0 else 0.0

    local_params = _get_model_params(model)
    debug_stats: Dict[str, float] | None = None
    if track_debug and n_steps > 0:
        debug_stats = {
            "delta_l2_before_step_mean": float(delta_l2_before_sum / n_steps),
            "delta_l2_after_step_mean": float(delta_l2_after_sum / n_steps),
            "prox_term_before_step_mean": float(prox_before_sum / n_steps),
            "prox_term_after_step_mean": float(prox_after_sum / n_steps),
        }
    if n_steps == 0:
        delta_l2_min = float("nan")
        delta_l2_max = float("nan")
        prox_term_min = float("nan")
        prox_term_max = float("nan")
    return (
        local_params - global_params,
        ce_loss_sum,
        prox_term_sum,
        delta_l2_sum,
        float(delta_l2_min),
        float(delta_l2_max),
        n_steps,
        float(prox_term_min),
        float(prox_term_max),
        int(prox_term_nan_count),
        int(prox_term_inf_count),
        debug_stats,
        step_debug_rows,
    )


def _eval_binary_dataset(
    x_data: np.ndarray,
    y_data: np.ndarray,
    params: np.ndarray,
    model_spec: SynthModelSpec,
    device: torch.device,
) -> tuple[float, float]:
    model = build_synth_model(model_spec).to(device)
    _set_model_params(model, params, device)
    model.eval()

    with torch.no_grad():
        x = torch.tensor(x_data, dtype=torch.float32, device=device)
        logits = model(x)
        if model_spec.output_dim == 1:
            criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss()
            y = torch.tensor(y_data, dtype=torch.float32, device=device).reshape(-1, 1)
            loss = float(criterion(logits, y).item())
            preds = (torch.sigmoid(logits) >= 0.5).float()
            acc = float((preds == y).float().mean().item())
        elif model_spec.output_dim == 2:
            criterion = torch.nn.CrossEntropyLoss()
            y = torch.tensor(y_data, dtype=torch.long, device=device).reshape(-1)
            loss = float(criterion(logits, y).item())
            preds = torch.argmax(logits, dim=1)
            acc = float((preds == y).float().mean().item())
        else:
            raise ValueError(f"Unsupported output_dim={model_spec.output_dim} for binary classification.")

    return loss, acc


def _eval_per_client_val_acc_mean(
    clients: Sequence[ClientDataset],
    params: np.ndarray,
    model_spec: SynthModelSpec,
    device: torch.device,
) -> float:
    accs = []
    for client in clients:
        if int(np.asarray(client.y_val).size) == 0:
            continue
        _, acc = _eval_binary_dataset(client.x_val, client.y_val, params, model_spec, device)
        accs.append(float(acc))
    if not accs:
        return float("nan")
    return float(np.mean(np.asarray(accs, dtype=np.float64)))


def run_synth2d_engine(
    clients: Sequence[ClientDataset],
    x_global_val: np.ndarray | None,
    y_global_val: np.ndarray | None,
    x_test: np.ndarray | None,
    y_test: np.ndarray | None,
    dataset_debug: Dict[str, Any] | None,
    model_cfg: Dict[str, Any] | None,
    rounds: int,
    sample_fraction: float,
    local_epochs: int,
    batch_size: int,
    lr: float,
    fl_method: str,
    fedprox_mu: float,
    optimizer_name: str,
    aggregator,
    scorer,
    device_name: str,
    seed: int = 0,
    debug_fedprox: bool = False,
    attack_mode: str = "label_flip",
    attack_scale: float = 1.0,
) -> RunOutputs:
    if not clients:
        raise ValueError("Expected non-empty clients list.")
    if x_global_val is None or y_global_val is None:
        raise ValueError("run_synth2d_engine requires held-out global x_global_val and y_global_val.")
    if int(np.asarray(y_global_val).size) == 0:
        raise ValueError("Held-out global validation set must be non-empty.")
    if x_test is None or y_test is None:
        raise ValueError("run_synth2d_engine requires held-out global x_test and y_test.")
    if int(np.asarray(y_test).size) == 0:
        raise ValueError("Held-out global test set must be non-empty.")

    n_clients = len(clients)
    m_per_round = max(1, int(round(sample_fraction * n_clients)))
    m_per_round = min(m_per_round, n_clients)

    rng = np.random.default_rng(seed)
    device = _resolve_device(device_name)
    method = str(fl_method).strip().lower()
    if method not in {"fedavg", "fedprox"}:
        raise ValueError(f"Unsupported fl.method: {fl_method}. Expected one of: fedavg, fedprox")
    mu = float(fedprox_mu)
    if mu < 0.0:
        raise ValueError("fl.fedprox_mu must be >= 0.0")
    attack_mode_norm = str(attack_mode).strip().lower()
    if attack_mode_norm not in {"label_flip", "byzantine_signflip", "signflip_scaled"}:
        raise ValueError("attack_mode must be one of: label_flip, byzantine_signflip, signflip_scaled")
    attack_scale_val = float(attack_scale)
    if attack_scale_val < 1.0:
        raise ValueError("attack_scale must be >= 1.0")
    model_spec = resolve_synth_model_spec(model_cfg)
    global_params = _init_global_params(model_spec, device)
    global_label_hist = _label_hist_binary(
        np.concatenate([np.asarray(client.y_train, dtype=np.int64) for client in clients], axis=0)
    )
    metric_rows = []
    score_rows = []
    fedprox_debug_rows = []
    fedprox_step_debug_rows = []
    start = time.time()

    for r in range(rounds):
        selected = rng.choice(n_clients, size=m_per_round, replace=False)
        selected_clients = [clients[int(cid)] for cid in selected.tolist()]

        client_updates = []
        train_sizes = []
        round_ce_loss_sum = 0.0
        round_prox_term_sum = 0.0
        round_delta_l2_sum = 0.0
        round_delta_l2_min = float("inf")
        round_delta_l2_max = float("-inf")
        round_steps = 0
        round_prox_term_min = float("inf")
        round_prox_term_max = float("-inf")
        round_prox_term_nan_count = 0
        round_prox_term_inf_count = 0
        round_debug_recorded = False
        debug_client_id = 0
        for client in selected_clients:
            (
                update,
                ce_loss_sum,
                prox_term_sum,
                delta_l2_sum,
                delta_l2_min,
                delta_l2_max,
                n_steps,
                prox_term_min,
                prox_term_max,
                prox_term_nan_count,
                prox_term_inf_count,
                local_debug,
                step_debug_rows,
            ) = _train_local_update(
                client=client,
                global_params=global_params,
                model_spec=model_spec,
                local_epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                fl_method=method,
                fedprox_mu=mu,
                optimizer_name=optimizer_name,
                device=device,
                track_debug=(int(client.client_id) == debug_client_id),
                track_step_debug=bool(debug_fedprox),
                round_idx=int(r),
            )
            if bool(client.is_attacker) and attack_mode_norm in {"byzantine_signflip", "signflip_scaled"}:
                update = (-attack_scale_val * np.asarray(update, dtype=np.float64)).astype(np.float64, copy=False)
            client_updates.append(update)
            train_sizes.append(int(client.y_train.shape[0]))
            round_ce_loss_sum += float(ce_loss_sum)
            round_prox_term_sum += float(prox_term_sum)
            round_delta_l2_sum += float(delta_l2_sum)
            round_delta_l2_min = min(round_delta_l2_min, float(delta_l2_min))
            round_delta_l2_max = max(round_delta_l2_max, float(delta_l2_max))
            round_steps += int(n_steps)
            round_prox_term_min = min(round_prox_term_min, float(prox_term_min))
            round_prox_term_max = max(round_prox_term_max, float(prox_term_max))
            round_prox_term_nan_count += int(prox_term_nan_count)
            round_prox_term_inf_count += int(prox_term_inf_count)
            if step_debug_rows:
                fedprox_step_debug_rows.extend(step_debug_rows)
            if local_debug is not None:
                fedprox_debug_rows.append(
                    {
                        "round": r,
                        "client_id": int(client.client_id),
                        "selected": True,
                        "n_steps": int(n_steps),
                        "fl_method": method,
                        "fedprox_mu": float(mu),
                        "delta_l2_before_step_mean": float(local_debug["delta_l2_before_step_mean"]),
                        "delta_l2_after_step_mean": float(local_debug["delta_l2_after_step_mean"]),
                        "prox_term_before_step_mean": float(local_debug["prox_term_before_step_mean"]),
                        "prox_term_after_step_mean": float(local_debug["prox_term_after_step_mean"]),
                    }
                )
                round_debug_recorded = True

        if not round_debug_recorded:
            fedprox_debug_rows.append(
                {
                    "round": r,
                    "client_id": int(debug_client_id),
                    "selected": False,
                    "n_steps": 0,
                    "fl_method": method,
                    "fedprox_mu": float(mu),
                    "delta_l2_before_step_mean": float("nan"),
                    "delta_l2_after_step_mean": float("nan"),
                    "prox_term_before_step_mean": float("nan"),
                    "prox_term_after_step_mean": float("nan"),
                }
            )

        updates = np.stack(client_updates, axis=0).astype(np.float64, copy=False)
        size_weights = np.asarray(train_sizes, dtype=np.float64)

        sctx = ScoringContext(
            round_idx=r,
            meta={
                "selected_client_ids": selected.tolist(),
                "clients": selected_clients,
                "global_params": global_params.copy(),
                "global_predict_proba": (
                    lambda x_data, _p=global_params.copy(), _spec=model_spec, _device=device:
                    _predict_binary_proba(
                        x_data=np.asarray(x_data, dtype=np.float32),
                        params=_p,
                        model_spec=_spec,
                        device=_device,
                    )
                ),
                "eval_loss_at_params": (
                    lambda params_vec, x_data, y_data, _spec=model_spec, _device=device:
                    _eval_binary_dataset(
                        x_data=np.asarray(x_data, dtype=np.float32),
                        y_data=np.asarray(y_data, dtype=np.int64),
                        params=np.asarray(params_vec, dtype=np.float64),
                        model_spec=_spec,
                        device=_device,
                    )[0]
                ),
                "x_global_val": np.asarray(x_global_val, dtype=np.float32),
                "y_global_val": np.asarray(y_global_val, dtype=np.int64),
                "global_label_hist": global_label_hist.copy(),
            },
        )
        scorer.fit_round(updates, sctx)
        raw_scores = np.asarray(scorer.score_clients(updates, sctx), dtype=np.float64).reshape(-1)
        if raw_scores.shape[0] != updates.shape[0]:
            raise AssertionError(
                f"Scorer returned shape {raw_scores.shape}, expected ({updates.shape[0]},)."
            )
        score_semantics = str(
            sctx.meta.get("score_semantics", getattr(scorer, "score_semantics", "reliability"))
        ).lower()
        harm_scores, reliability = _resolve_harm_and_reliability_from_meta(raw_scores, score_semantics, sctx.meta)
        if not np.all(np.isfinite(reliability)):
            raise AssertionError("Non-finite reliability from scorer.")
        global_val_loss_before_score = _extract_optional_meta_scalar(sctx.meta, "global_val_loss_before")
        global_val_loss_after_i = _extract_optional_meta_vector_or_nan(
            sctx.meta, "global_val_loss_after_i", len(selected_clients)
        )
        global_val_delta_loss_i = _extract_optional_meta_vector_or_nan(
            sctx.meta, "global_val_delta_loss_i", len(selected_clients)
        )
        if str(getattr(scorer, "name", "")).strip().lower() == "global_val_harm":
            if not np.isfinite(global_val_loss_before_score):
                raise AssertionError("global_val_harm scorer produced non-finite global_val_loss_before.")
            if not np.all(np.isfinite(global_val_loss_after_i)):
                raise AssertionError("global_val_harm scorer produced non-finite global_val_loss_after_i.")
            if not np.all(np.isfinite(global_val_delta_loss_i)):
                raise AssertionError("global_val_harm scorer produced non-finite global_val_delta_loss_i.")

        clean_scores = []
        noisy_scores = []
        attack_scores = []
        for rel, client in zip(reliability.tolist(), selected_clients):
            role = _client_role(client)
            if role == "clean":
                clean_scores.append(float(rel))
            elif role == "noisy":
                noisy_scores.append(float(rel))
            elif role == "attacker":
                attack_scores.append(float(rel))
        score_mean_clean = float(np.mean(clean_scores)) if clean_scores else float("nan")
        score_mean_noisy = float(np.mean(noisy_scores)) if noisy_scores else float("nan")
        score_mean_attack = float(np.mean(attack_scores)) if attack_scores else float("nan")

        scorer_weights = _extract_scorer_weights(sctx.meta, updates.shape[0])
        if scorer_weights is not None:
            weights = scorer_weights
        else:
            weights = size_weights * reliability
            if float(weights.sum()) <= 1e-12:
                weights = np.ones_like(weights, dtype=np.float64)

        actx = AggregationContext(
            round_idx=r,
            meta={
                "selected_client_ids": selected.tolist(),
                "score_semantics": score_semantics,
            },
        )
        global_update = aggregator.aggregate(updates, weights=weights, ctx=actx)
        weight_i_used = _resolve_weight_i_used(actx.meta, len(selected_clients), fallback_weights=weights)
        cos_i = _cosine_to_aggregate_update(updates, global_update)
        if not np.all(np.isfinite(cos_i)):
            raise AssertionError("Found non-finite cos_i values.")
        if not np.all(np.isfinite(weight_i_used)):
            raise AssertionError("Found non-finite weight_i values.")
        if not np.all(np.isfinite(reliability)):
            raise AssertionError("Found non-finite reliability values.")
        if bool(actx.meta.get("weights_sum_to_one", True)):
            if not np.isclose(float(np.sum(weight_i_used)), 1.0, atol=1e-6):
                raise AssertionError("Per-round weight_i values do not sum to 1.")
        global_params = global_params + global_update

        global_val_loss, global_val_acc = _eval_binary_dataset(
            x_global_val, y_global_val, global_params, model_spec, device
        )
        per_client_val_acc_mean = _eval_per_client_val_acc_mean(clients, global_params, model_spec, device)
        test_loss, test_acc = _eval_binary_dataset(x_test, y_test, global_params, model_spec, device)
        n_local_steps_total = int(round_steps)
        delta_l2_mean = float(round_delta_l2_sum / round_steps) if round_steps > 0 else float("nan")
        delta_l2_min = float(round_delta_l2_min) if round_steps > 0 else float("nan")
        delta_l2_max = float(round_delta_l2_max) if round_steps > 0 else float("nan")
        ce_loss_mean = float(round_ce_loss_sum / round_steps) if round_steps > 0 else float("nan")
        prox_term_mean = float(round_prox_term_sum / round_steps) if round_steps > 0 else float("nan")
        prox_term_min = float(round_prox_term_min) if round_steps > 0 else float("nan")
        prox_term_max = float(round_prox_term_max) if round_steps > 0 else float("nan")
        if method == "fedprox":
            assert n_local_steps_total > 0 and delta_l2_mean > 0.0, (
                "FedProx delta_l2 is zero; p0 snapshot or local update wiring is broken."
            )
        metric_rows.append(
            {
                "round": r,
                "fl_method": method,
                "fl.method": method,
                "fedprox_mu": float(mu),
                "n_local_steps_total": n_local_steps_total,
                "delta_l2_mean": delta_l2_mean,
                "delta_l2_min": delta_l2_min,
                "delta_l2_max": delta_l2_max,
                "ce_loss_mean": ce_loss_mean,
                "prox_term_mean": prox_term_mean,
                "prox_term_min": prox_term_min,
                "prox_term_max": prox_term_max,
                "prox_nan_count": int(round_prox_term_nan_count),
                "prox_inf_count": int(round_prox_term_inf_count),
                "prox_term_nan_count": int(round_prox_term_nan_count),
                "prox_term_inf_count": int(round_prox_term_inf_count),
                "score_mean_clean": score_mean_clean,
                "score_mean_noisy": score_mean_noisy,
                "score_mean_attack": score_mean_attack,
                "val_loss": float(global_val_loss),
                "val_acc": float(global_val_acc),
                "global_val_loss": float(global_val_loss),
                "global_val_acc": float(global_val_acc),
                "client_val_acc_mean": float(per_client_val_acc_mean),
                "per_client_val_acc_mean": float(per_client_val_acc_mean),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "update_l2": float(np.linalg.norm(global_update)),
            }
        )
        for i, (cid, sc, hs, rel, client) in enumerate(
            zip(
                selected.tolist(),
                raw_scores.tolist(),
                harm_scores.tolist(),
                reliability.tolist(),
                selected_clients,
            )
        ):
            score_rows.append(
                {
                    "round": r,
                    "client_id": int(cid),
                    "score": float(sc),
                    "harm_score": float(hs),
                    "reliability": float(rel),
                    "global_val_loss_before": float(global_val_loss_before_score),
                    "global_val_loss_after": float(global_val_loss_after_i[i]),
                    "global_val_delta_loss": float(global_val_delta_loss_i[i]),
                    "client_role": _client_role(client),
                    "cos_i": float(cos_i[i]),
                    "weight_i": float(weight_i_used[i]),
                }
            )

    elapsed = time.time() - start
    metrics_round = pd.DataFrame(metric_rows)
    client_scores_round = pd.DataFrame(score_rows)
    fedprox_debug_round = pd.DataFrame(fedprox_debug_rows)
    fedprox_debug = pd.DataFrame(fedprox_step_debug_rows) if debug_fedprox else None
    summary = {
        "rounds": int(rounds),
        "n_clients": int(n_clients),
        "dim": int(global_params.shape[0]),
        "model": model_spec.name,
        "device": str(device),
        "fl_method": method,
        "fl.method": method,
        "fedprox_mu": float(mu),
        "n_global_val": int(np.asarray(y_global_val).size),
        "global_val_pos_rate": float(np.mean(np.asarray(y_global_val) == 1)),
        "n_test": int(np.asarray(y_test).size),
        "test_pos_rate": float(np.mean(np.asarray(y_test) == 1)),
        "final_metric_name": "final_test_acc",
        "final_test_acc": float(metrics_round["test_acc"].iloc[-1]),
        "final_test_loss": float(metrics_round["test_loss"].iloc[-1]),
        "final_val_loss": float(metrics_round["val_loss"].iloc[-1]),
        "final_val_acc": float(metrics_round["val_acc"].iloc[-1]),
        "global_val_loss": float(metrics_round["global_val_loss"].iloc[-1]),
        "global_val_acc": float(metrics_round["global_val_acc"].iloc[-1]),
        "n_local_steps_total": int(metrics_round["n_local_steps_total"].iloc[-1]),
        "delta_l2_mean": float(metrics_round["delta_l2_mean"].iloc[-1]),
        "delta_l2_min": float(metrics_round["delta_l2_min"].iloc[-1]),
        "delta_l2_max": float(metrics_round["delta_l2_max"].iloc[-1]),
        "ce_loss_mean": float(metrics_round["ce_loss_mean"].iloc[-1]),
        "prox_term_mean": float(metrics_round["prox_term_mean"].iloc[-1]),
        "prox_term_min": float(metrics_round["prox_term_min"].iloc[-1]),
        "prox_term_max": float(metrics_round["prox_term_max"].iloc[-1]),
        "prox_nan_count": int(metrics_round["prox_nan_count"].iloc[-1]),
        "prox_inf_count": int(metrics_round["prox_inf_count"].iloc[-1]),
        "prox_term_nan_count": int(metrics_round["prox_term_nan_count"].iloc[-1]),
        "prox_term_inf_count": int(metrics_round["prox_term_inf_count"].iloc[-1]),
        "score_mean_clean": float(metrics_round["score_mean_clean"].iloc[-1]),
        "score_mean_noisy": float(metrics_round["score_mean_noisy"].iloc[-1]),
        "score_mean_attack": float(metrics_round["score_mean_attack"].iloc[-1]),
        "client_val_acc_mean": float(metrics_round["client_val_acc_mean"].iloc[-1]),
        "per_client_val_acc_mean": float(metrics_round["per_client_val_acc_mean"].iloc[-1]),
        "runtime_sec": float(elapsed),
        "num_noisy_clients": int(sum(int(c.is_noisy) for c in clients)),
        "num_attack_clients": int(sum(int(c.is_attacker) for c in clients)),
        "attack_mode": "byzantine_signflip" if attack_mode_norm == "signflip_scaled" else attack_mode_norm,
        "attack_scale": float(attack_scale_val),
    }
    if dataset_debug:
        summary.update(dataset_debug)
    return RunOutputs(
        metrics_round=metrics_round,
        summary=summary,
        client_scores_round=client_scores_round,
        fedprox_debug_round=fedprox_debug_round,
        fedprox_debug=fedprox_debug,
    )
