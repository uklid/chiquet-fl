from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


@dataclass
class ClientDataset:
    client_id: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    is_noisy: bool = False
    is_attacker: bool = False


@dataclass
class SyntheticDataBundle:
    clients: list[ClientDataset]
    x_global_val: np.ndarray
    y_global_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    debug: dict[str, Any]


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _client_counts(total: int, num_clients: int, quantity_skew: float, rng: np.random.Generator) -> np.ndarray:
    min_required = 2 * num_clients
    if total < min_required:
        raise ValueError(
            f"n_samples_total ({total}) must be >= {min_required} to create train/val splits per client."
        )

    base = np.full((num_clients,), 2, dtype=np.int64)
    remaining = total - int(base.sum())
    if remaining <= 0:
        return base

    if quantity_skew <= 1e-12:
        extra = np.full((num_clients,), remaining // num_clients, dtype=np.int64)
        extra[: remaining % num_clients] += 1
        return base + extra

    alpha = max(0.05, (1.0 - quantity_skew) * 8.0 + 0.2)
    probs = rng.dirichlet(np.full((num_clients,), alpha, dtype=np.float64))
    extra = rng.multinomial(remaining, probs)
    return base + extra


def _class1_props(num_clients: int, label_skew: float, rng: np.random.Generator) -> np.ndarray:
    if label_skew <= 1e-12:
        return np.full((num_clients,), 0.5, dtype=np.float64)
    alpha = max(0.05, (1.0 - label_skew) * 12.0 + 0.2)
    props = rng.beta(alpha, alpha, size=num_clients)
    return np.clip(props, 0.01, 0.99)


def _allocate_class1_counts(
    counts: np.ndarray,
    class1_total: int,
    class1_props: np.ndarray,
) -> np.ndarray:
    raw = counts.astype(np.float64) * class1_props.astype(np.float64)
    class1 = np.floor(raw).astype(np.int64)
    class1 = np.minimum(class1, counts)

    need = int(class1_total - class1.sum())
    if need > 0:
        order = np.argsort(-(raw - class1))
        for idx in order:
            if need == 0:
                break
            cap = int(counts[idx] - class1[idx])
            if cap <= 0:
                continue
            add = min(cap, need)
            class1[idx] += add
            need -= add
    elif need < 0:
        need = -need
        order = np.argsort(raw - class1)
        for idx in order:
            if need == 0:
                break
            rem = int(class1[idx])
            if rem <= 0:
                continue
            sub = min(rem, need)
            class1[idx] -= sub
            need -= sub

    # Final strict fix-up to guarantee exact class1_total.
    current = int(class1.sum())
    if current != class1_total:
        delta = class1_total - current
        if delta > 0:
            order = np.argsort(-(counts - class1))
            for idx in order:
                if delta == 0:
                    break
                cap = int(counts[idx] - class1[idx])
                if cap <= 0:
                    continue
                add = min(cap, delta)
                class1[idx] += add
                delta -= add
        else:
            delta = -delta
            order = np.argsort(-class1)
            for idx in order:
                if delta == 0:
                    break
                rem = int(class1[idx])
                if rem <= 0:
                    continue
                sub = min(rem, delta)
                class1[idx] -= sub
                delta -= sub

    if int(class1.sum()) != class1_total:
        raise RuntimeError("Failed to allocate class counts exactly.")
    if np.any(class1 < 0) or np.any(class1 > counts):
        raise RuntimeError("Invalid class allocation produced.")
    return class1


def _parse_samples_per_client(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {"", "none", "null"}:
        return None
    val = int(raw)
    if val <= 0:
        raise ValueError("samples_per_client must be > 0 when provided.")
    return val


def _make_2d_binary(
    n_samples: int,
    class_sep: float,
    flip_y: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    x, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=random_state,
    )
    return x.astype(np.float32), y.astype(np.int64)


def _split_train_val_stratified(
    x_client: np.ndarray,
    y_client: np.ndarray,
    val_frac: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_client = int(y_client.size)
    if n_client < 2:
        raise ValueError("Each client must have at least 2 samples for train/val split.")

    n_val = max(1, int(round(n_client * val_frac)))
    n_val = min(n_val, n_client - 1)

    # For binary/multiclass stratification, force enough slots for each class when feasible.
    classes, class_counts = np.unique(y_client, return_counts=True)
    n_classes = int(classes.size)
    if n_classes > 1:
        min_val = n_classes
        max_val = n_client - n_classes
        if max_val >= min_val:
            n_val = int(np.clip(n_val, min_val, max_val))

    # Prefer stratified split; fallback only when class support is insufficient.
    try:
        x_train, x_val, y_train, y_val = train_test_split(
            x_client,
            y_client,
            test_size=n_val,
            random_state=split_seed,
            shuffle=True,
            stratify=y_client,
        )
    except ValueError:
        x_train, x_val, y_train, y_val = train_test_split(
            x_client,
            y_client,
            test_size=n_val,
            random_state=split_seed,
            shuffle=True,
            stratify=None,
        )

    return (
        x_train.astype(np.float32, copy=False),
        y_train.astype(np.int64, copy=False),
        x_val.astype(np.float32, copy=False),
        y_val.astype(np.int64, copy=False),
    )


def build_synthetic_data(cfg: Any) -> SyntheticDataBundle:
    num_clients = int(_cfg_get(cfg, "num_clients", 10))
    n_samples_total = int(_cfg_get(cfg, "n_samples_total", 2000))
    samples_per_client = _parse_samples_per_client(_cfg_get(cfg, "samples_per_client", None))
    quantity_skew = float(_cfg_get(cfg, "quantity_skew", 0.0))
    label_skew = float(_cfg_get(cfg, "label_skew", 0.0))
    class_sep = float(_cfg_get(cfg, "class_sep", 1.5))
    flip_y = float(_cfg_get(cfg, "flip_y", 0.0))
    label_noise_rate = float(_cfg_get(cfg, "label_noise_rate", 0.0))
    noise_clients_frac = float(_cfg_get(cfg, "noise_clients_frac", 0.0))
    noise_rate = float(_cfg_get(cfg, "noise_rate", 0.0))
    attack_clients_frac = float(_cfg_get(cfg, "attack_clients_frac", 0.0))
    attack_mode = str(_cfg_get(cfg, "attack_mode", "label_flip")).strip().lower()
    attack_scale = float(_cfg_get(cfg, "attack_scale", 1.0))
    val_frac = float(_cfg_get(cfg, "val_frac", 0.15))
    raw_n_global_val_samples = _cfg_get(cfg, "n_global_val_samples", None)
    raw_n_test_samples = _cfg_get(cfg, "n_test_samples", None)
    standardize_mode = str(_cfg_get(cfg, "standardize_mode", "global_train_only")).strip().lower()
    seed = int(_cfg_get(cfg, "seed", 0))

    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")
    if samples_per_client is not None:
        n_samples_total = int(num_clients * samples_per_client)
    if raw_n_global_val_samples is None:
        n_global_val_samples = max(200, n_samples_total // 4)
    else:
        n_global_val_samples = int(raw_n_global_val_samples)
    if raw_n_test_samples is None:
        n_test_samples = max(200, n_samples_total // 4)
    else:
        n_test_samples = int(raw_n_test_samples)
    if n_samples_total <= 0:
        raise ValueError("n_samples_total must be > 0")
    if not (0.0 <= quantity_skew <= 1.0):
        raise ValueError("quantity_skew must be in [0, 1]")
    if not (0.0 <= label_skew <= 1.0):
        raise ValueError("label_skew must be in [0, 1]")
    if class_sep <= 0.0:
        raise ValueError("class_sep must be > 0")
    if not (0.0 <= flip_y <= 1.0):
        raise ValueError("flip_y must be in [0, 1]")
    if not (0.0 <= label_noise_rate <= 1.0):
        raise ValueError("label_noise_rate must be in [0, 1]")
    if not (0.0 <= noise_clients_frac <= 1.0):
        raise ValueError("noise_clients_frac must be in [0, 1]")
    if not (0.0 <= attack_clients_frac <= 1.0):
        raise ValueError("attack_clients_frac must be in [0, 1]")
    if attack_mode not in {"label_flip", "byzantine_signflip", "signflip_scaled"}:
        raise ValueError("attack_mode must be one of: label_flip, byzantine_signflip, signflip_scaled")
    if attack_scale < 1.0:
        raise ValueError("attack_scale must be >= 1.0")
    if not (0.0 <= noise_rate <= 1.0):
        raise ValueError("noise_rate must be in [0, 1]")
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must be in (0, 1)")
    if n_global_val_samples <= 0:
        raise ValueError("n_global_val_samples must be > 0")
    if n_test_samples <= 0:
        raise ValueError("n_test_samples must be > 0")
    if standardize_mode not in {"global_train_only", "none"}:
        raise ValueError("standardize_mode must be one of: global_train_only, none")

    # Single RNG stream for all stochastic components in synthetic generation.
    rng = np.random.default_rng(seed)
    cls_seed_train = int(rng.integers(0, np.iinfo(np.int32).max))
    cls_seed_global_val = int(rng.integers(0, np.iinfo(np.int32).max))
    cls_seed_test = int(rng.integers(0, np.iinfo(np.int32).max))

    x, y = _make_2d_binary(
        n_samples=n_samples_total,
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=cls_seed_train,
    )
    x_global_val, y_global_val = _make_2d_binary(
        n_samples=n_global_val_samples,
        class_sep=class_sep,
        flip_y=0.0,  # Keep held-out global val labels clean.
        random_state=cls_seed_global_val,
    )
    x_test, y_test = _make_2d_binary(
        n_samples=n_test_samples,
        class_sep=class_sep,
        flip_y=0.0,  # Keep held-out global test labels clean.
        random_state=cls_seed_test,
    )

    if label_noise_rate > 0.0:
        flip_mask = rng.random(y.size) < label_noise_rate
        y = np.where(flip_mask, 1 - y, y).astype(np.int64, copy=False)

    counts = _client_counts(n_samples_total, num_clients, quantity_skew, rng)
    class1_props = _class1_props(num_clients, label_skew, rng)

    idx_class0 = np.where(y == 0)[0]
    idx_class1 = np.where(y == 1)[0]
    rng.shuffle(idx_class0)
    rng.shuffle(idx_class1)

    class1_counts = _allocate_class1_counts(
        counts=counts,
        class1_total=int(idx_class1.size),
        class1_props=class1_props,
    )
    class0_counts = counts - class1_counts

    n_attack = min(num_clients, int(round(num_clients * attack_clients_frac)))
    all_clients = np.arange(num_clients, dtype=np.int64)
    attack_ids = set(rng.choice(all_clients, size=n_attack, replace=False).tolist()) if n_attack > 0 else set()

    remaining = np.array([cid for cid in all_clients.tolist() if cid not in attack_ids], dtype=np.int64)
    n_noisy = min(remaining.size, int(round(num_clients * noise_clients_frac)))
    noise_ids = set(rng.choice(remaining, size=n_noisy, replace=False).tolist()) if n_noisy > 0 else set()

    clients: list[ClientDataset] = []
    c0_ptr = 0
    c1_ptr = 0
    noisy_train_label_flips = 0
    attack_train_label_flips = 0

    for cid in range(num_clients):
        n0 = int(class0_counts[cid])
        n1 = int(class1_counts[cid])
        chosen = np.concatenate(
            [
                idx_class0[c0_ptr : c0_ptr + n0],
                idx_class1[c1_ptr : c1_ptr + n1],
            ],
            axis=0,
        )
        c0_ptr += n0
        c1_ptr += n1

        x_client = x[chosen]
        y_client = y[chosen].copy()

        split_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        x_train, y_train, x_val, y_val = _split_train_val_stratified(
            x_client=x_client,
            y_client=y_client,
            val_frac=val_frac,
            split_seed=split_seed,
        )

        is_attacker = cid in attack_ids
        is_noisy = cid in noise_ids

        if is_attacker:
            if attack_mode == "label_flip":
                attack_train_label_flips += int(y_train.size)
                y_train = 1 - y_train
            elif attack_mode in {"byzantine_signflip", "signflip_scaled"}:
                # Keep attacker training labels unchanged; poisoning is applied on
                # model updates in the FL engine.
                pass
        elif is_noisy and y_train.size > 0:
            flip_mask = rng.random(y_train.size) < noise_rate
            noisy_train_label_flips += int(flip_mask.sum())
            y_train = np.where(flip_mask, 1 - y_train, y_train)

        clients.append(
            ClientDataset(
                client_id=cid,
                x_train=x_train,
                y_train=y_train.astype(np.int64, copy=False),
                x_val=x_val,
                y_val=y_val,
                is_noisy=is_noisy,
                is_attacker=is_attacker,
            )
        )

    # Provenance: fit scaler on train only (never val/test/global test).
    standardize_enabled = standardize_mode != "none"
    scaler_fit_scope = "global_train"
    n_global_val_used_for_scaler = 0
    n_global_test_used_for_scaler = 0
    if standardize_enabled:
        x_train_all = np.concatenate([client.x_train for client in clients], axis=0)
        x_mean = x_train_all.mean(axis=0, keepdims=True)
        x_std = x_train_all.std(axis=0, keepdims=True) + 1e-6
        scaler_fit_n = int(x_train_all.shape[0])
        for client in clients:
            client.x_train = ((client.x_train - x_mean) / x_std).astype(np.float32, copy=False)
            client.x_val = ((client.x_val - x_mean) / x_std).astype(np.float32, copy=False)
        x_global_val = ((x_global_val - x_mean) / x_std).astype(np.float32, copy=False)
        x_test = ((x_test - x_mean) / x_std).astype(np.float32, copy=False)
    else:
        scaler_fit_n = 0
        standardize_mode = "none"
        scaler_fit_scope = "global_train"
        x_global_val = x_global_val.astype(np.float32, copy=False)
        x_test = x_test.astype(np.float32, copy=False)

    if scaler_fit_scope == "train_plus_test":
        raise RuntimeError("Invalid scaler provenance: scaler_fit_scope must never be train_plus_test.")
    if n_global_val_used_for_scaler != 0:
        raise RuntimeError("Invalid scaler provenance: n_global_val_used_for_scaler must be 0.")
    if n_global_test_used_for_scaler != 0:
        raise RuntimeError("Invalid scaler provenance: n_global_test_used_for_scaler must be 0.")

    per_client_train_size = [int(client.y_train.size) for client in clients]
    per_client_train_pos_rate = [
        float(np.mean(client.y_train == 1)) if client.y_train.size > 0 else 0.0
        for client in clients
    ]
    per_client_val_size = [int(client.y_val.size) for client in clients]
    per_client_val_pos_rate = [
        float(np.mean(client.y_val == 1)) if client.y_val.size > 0 else 0.0
        for client in clients
    ]
    y_train_all = np.concatenate([client.y_train for client in clients], axis=0)
    y_val_all = np.concatenate([client.y_val for client in clients], axis=0)

    debug = {
        "n_global_val": int(y_global_val.size),
        "global_val_pos_rate": float(np.mean(y_global_val == 1)),
        "n_test": int(y_test.size),
        "test_pos_rate": float(np.mean(y_test == 1)),
        "n_clients": int(num_clients),
        "n_noisy_clients": int(len(noise_ids)),
        "n_attack_clients": int(len(attack_ids)),
        "per_client_train_size": per_client_train_size,
        "per_client_train_pos_rate": per_client_train_pos_rate,
        "per_client_val_size": per_client_val_size,
        "per_client_val_pos_rate": per_client_val_pos_rate,
        "global_train_pos_rate": float(np.mean(y_train_all == 1)),
        "client_val_pool_pos_rate": float(np.mean(y_val_all == 1)),
        "feature_standardized": bool(standardize_enabled),
        "standardize_enabled": bool(standardize_enabled),
        "standardize_mode": standardize_mode,
        "scaler_fit_scope": scaler_fit_scope,
        "scaler_fit_n": int(scaler_fit_n),
        "n_global_val_used_for_scaler": int(n_global_val_used_for_scaler),
        "n_global_test_used_for_scaler": int(n_global_test_used_for_scaler),
        "noise_client_ids": sorted(int(i) for i in noise_ids),
        "attack_client_ids": sorted(int(i) for i in attack_ids),
        "n_client_train_labels_flipped_noisy": int(noisy_train_label_flips),
        "n_client_train_labels_flipped_attack": int(attack_train_label_flips),
        "n_client_val_labels_flipped": 0,
        "n_global_val_labels_flipped": 0,
        "n_global_test_labels_flipped": 0,
        "label_flip_scope": (
            "selected_client_train_only" if attack_mode == "label_flip" else "noise_clients_train_only"
        ),
        "global_val_label_flip_applied": False,
        "global_test_label_flip_applied": False,
        "attack_mode": ("byzantine_signflip" if attack_mode == "signflip_scaled" else attack_mode),
        "attack_scale": float(attack_scale),
    }

    return SyntheticDataBundle(
        clients=clients,
        x_global_val=x_global_val.astype(np.float32, copy=False),
        y_global_val=y_global_val.astype(np.int64, copy=False),
        x_test=x_test.astype(np.float32, copy=False),
        y_test=y_test.astype(np.int64, copy=False),
        debug=debug,
    )


def build_synthetic_clients(cfg: Any) -> list[ClientDataset]:
    return build_synthetic_data(cfg).clients
