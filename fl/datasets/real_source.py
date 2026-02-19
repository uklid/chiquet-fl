from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .synthetic import ClientDataset, SyntheticDataBundle


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _normalize_col_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name).strip() if ch.isalnum() or ch == "_")


def _resolve_col(
    df: pd.DataFrame,
    explicit: str | None,
    candidates: list[str] | None,
    kind: str,
) -> str:
    cols = [str(c) for c in df.columns]
    by_norm: dict[str, str] = {}
    for c in cols:
        by_norm.setdefault(_normalize_col_name(c), c)

    def _try(name: str | None) -> str | None:
        if name is None:
            return None
        raw = str(name)
        if raw in df.columns:
            return raw
        return by_norm.get(_normalize_col_name(raw))

    picked = _try(explicit)
    if picked is not None:
        return picked

    for c in candidates or []:
        picked = _try(c)
        if picked is not None:
            return picked

    preview = ", ".join(cols[:20])
    raise ValueError(f"Could not resolve {kind} column. Available columns (first 20): {preview}")


def _load_df(cfg: Any) -> pd.DataFrame:
    data_path = Path(str(_cfg_get(cfg, "data_path", ""))).expanduser()
    if not str(data_path):
        raise ValueError("dataset.params.data_path is required for real datasets.")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    file_format = str(_cfg_get(cfg, "file_format", "auto")).strip().lower()
    delimiter = _cfg_get(cfg, "delimiter", None)
    max_rows_raw = _cfg_get(cfg, "max_rows", None)
    max_rows = None if max_rows_raw is None else int(max_rows_raw)

    suffix = data_path.suffix.lower()
    if file_format == "auto":
        if suffix in {".csv", ".txt"}:
            file_format = "csv"
        elif suffix in {".parquet", ".pq"}:
            file_format = "parquet"
        else:
            file_format = "csv"

    if file_format == "csv":
        sep = "," if delimiter is None else str(delimiter)
        return pd.read_csv(data_path, sep=sep, nrows=max_rows, low_memory=False)
    if file_format == "parquet":
        df = pd.read_parquet(data_path)
        if max_rows is not None:
            df = df.iloc[:max_rows].copy()
        return df

    raise ValueError(f"Unsupported file_format: {file_format}")


def _to_binary_labels(
    y_raw: pd.Series,
    benign_labels: list[Any] | None,
    attack_labels: list[Any] | None,
) -> np.ndarray:
    vals = y_raw.to_numpy()
    vals_str = pd.Series(vals).astype(str).str.strip().str.lower().to_numpy()

    if attack_labels:
        attack_set = {str(v).strip().lower() for v in attack_labels}
        y = np.asarray([1 if v in attack_set else 0 for v in vals_str], dtype=np.int64)
        return y

    if benign_labels:
        benign_set = {str(v).strip().lower() for v in benign_labels}
        y = np.asarray([0 if v in benign_set else 1 for v in vals_str], dtype=np.int64)
        return y

    # Fallback: keep 0/1 numeric labels as-is; otherwise map known benign tokens to 0.
    y_num = pd.to_numeric(pd.Series(vals), errors="coerce")
    if y_num.notna().all():
        unique_vals = sorted(set(int(v) for v in y_num.to_numpy()))
        if set(unique_vals).issubset({0, 1}):
            return y_num.to_numpy(dtype=np.int64)

    benign_tokens = {"0", "benign", "normal", "background", "nonattack", "non-attack", "noattack"}
    y = np.asarray([0 if v in benign_tokens else 1 for v in vals_str], dtype=np.int64)
    return y


def _prepare_features(
    df: pd.DataFrame,
    label_col: str,
    source_col: str,
    drop_cols: list[str],
) -> pd.DataFrame:
    drop_set = {label_col, source_col, *drop_cols}
    cols = [c for c in df.columns if c not in drop_set]
    if not cols:
        raise ValueError("No feature columns remain after dropping label/source columns.")

    feat = df[cols].copy()
    for c in feat.columns:
        if pd.api.types.is_bool_dtype(feat[c]):
            feat[c] = feat[c].astype(np.int8)
    feat = pd.get_dummies(feat, dummy_na=True, dtype=np.float32)
    if feat.shape[1] == 0:
        raise ValueError("Feature matrix has zero columns after preprocessing.")
    return feat


def _split_global(
    x: pd.DataFrame,
    y: np.ndarray,
    source_ids: np.ndarray,
    test_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    n = int(y.shape[0])
    if n < 10:
        raise ValueError("Real dataset requires at least 10 rows.")
    if not (0.0 < test_frac < 1.0):
        raise ValueError("global_test_frac must be in (0,1).")
    if not (0.0 < val_frac < 1.0):
        raise ValueError("global_val_frac must be in (0,1).")
    if (test_frac + val_frac) >= 1.0:
        raise ValueError("global_test_frac + global_val_frac must be < 1.")

    idx = np.arange(n, dtype=np.int64)
    idx_train_val, idx_test = train_test_split(
        idx,
        test_size=test_frac,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    val_rel = val_frac / (1.0 - test_frac)
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=val_rel,
        random_state=seed + 1,
        shuffle=True,
        stratify=y[idx_train_val],
    )

    x_train = x.iloc[idx_train].reset_index(drop=True)
    y_train = y[idx_train].astype(np.int64, copy=False)
    src_train = source_ids[idx_train]
    x_val = x.iloc[idx_val].reset_index(drop=True)
    y_val = y[idx_val].astype(np.int64, copy=False)
    src_val = source_ids[idx_val]
    x_test = x.iloc[idx_test].reset_index(drop=True)
    y_test = y[idx_test].astype(np.int64, copy=False)
    return x_train, y_train, src_train, x_val, y_val, src_val, x_test, y_test


def _fillna_with_train_stats(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mode_norm = str(mode).strip().lower()
    if mode_norm in {"none", ""}:
        return x_train.fillna(0.0), x_val.fillna(0.0), x_test.fillna(0.0)
    if mode_norm in {"median", "train_median"}:
        med = x_train.median(axis=0, numeric_only=True)
        x_train = x_train.fillna(med).fillna(0.0)
        x_val = x_val.fillna(med).fillna(0.0)
        x_test = x_test.fillna(med).fillna(0.0)
        return x_train, x_val, x_test
    raise ValueError("dataset.params.fillna must be one of: none, median")


def _standardize_global_train_only(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "none":
        return (
            x_train.astype(np.float32, copy=False),
            x_val.astype(np.float32, copy=False),
            x_test.astype(np.float32, copy=False),
            {
                "feature_standardized": False,
                "standardize_enabled": False,
                "standardize_mode": "none",
                "scaler_fit_scope": "global_train",
                "scaler_fit_n": 0,
                "n_global_val_used_for_scaler": 0,
                "n_global_test_used_for_scaler": 0,
            },
        )

    if mode_norm != "global_train_only":
        raise ValueError("standardize_mode must be one of: global_train_only, none")

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = ((x_train - mean) / std).astype(np.float32, copy=False)
    x_val = ((x_val - mean) / std).astype(np.float32, copy=False)
    x_test = ((x_test - mean) / std).astype(np.float32, copy=False)
    return (
        x_train,
        x_val,
        x_test,
        {
            "feature_standardized": True,
            "standardize_enabled": True,
            "standardize_mode": "global_train_only",
            "scaler_fit_scope": "global_train",
            "scaler_fit_n": int(x_train.shape[0]),
            "n_global_val_used_for_scaler": 0,
            "n_global_test_used_for_scaler": 0,
        },
    )


_ROLE_BENIGN = "benign_heavy"
_ROLE_MIXED = "mixed"
_ROLE_ATTACK = "attack_heavy"
_ROLE_ORDER = [_ROLE_BENIGN, _ROLE_MIXED, _ROLE_ATTACK]


def _role_from_ratio(attack_ratio: float, low_thr: float, high_thr: float) -> str:
    r = float(attack_ratio)
    if r <= low_thr:
        return _ROLE_BENIGN
    if r >= high_thr:
        return _ROLE_ATTACK
    return _ROLE_MIXED


def _build_source_stats(
    src_train_arr: np.ndarray,
    y_train_all: np.ndarray,
    src_val_arr: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    train_s = pd.Series(src_train_arr.astype(str))
    val_s = pd.Series(src_val_arr.astype(str))
    train_counts = train_s.value_counts()
    val_counts = val_s.value_counts()
    out: dict[str, dict[str, float | int]] = {}
    for src, cnt in train_counts.items():
        src_str = str(src)
        tr_idx = np.where(src_train_arr == src)[0]
        attack_ratio = float(np.mean(y_train_all[tr_idx] == 1)) if tr_idx.size > 0 else 0.0
        out[src_str] = {
            "n_train": int(cnt),
            "n_val": int(val_counts.get(src_str, 0)),
            "attack_ratio": float(attack_ratio),
        }
    return out


def _normalize_targets(n_clients: int, per_bin_raw: list[int] | None) -> dict[str, int]:
    if n_clients <= 0:
        raise ValueError("partition n_clients must be > 0")

    if per_bin_raw is None:
        base = n_clients // 3
        rem = n_clients - (3 * base)
        vals = [base, base, base]
        vals[1] += rem
    else:
        vals = [int(x) for x in per_bin_raw]
        if len(vals) != 3:
            raise ValueError("partition.params.per_bin must have exactly 3 values [benign,mixed,attack].")
        if any(v < 0 for v in vals):
            raise ValueError("partition.params.per_bin values must be >= 0.")

    total = int(sum(vals))
    if total != n_clients:
        diff = n_clients - total
        # Adjust mixed bin first for minimum surprise.
        vals[1] = max(0, vals[1] + diff)
        total = int(sum(vals))
        if total != n_clients:
            # Final correction to attack-heavy bin.
            vals[2] = max(0, vals[2] + (n_clients - total))
    if int(sum(vals)) != n_clients:
        raise ValueError("Could not reconcile per_bin counts with n_clients.")
    return {_ROLE_BENIGN: vals[0], _ROLE_MIXED: vals[1], _ROLE_ATTACK: vals[2]}


def _enforce_min_mixed_target(targets: dict[str, int], min_mixed_clients: int) -> dict[str, int]:
    out = dict(targets)
    min_mixed = max(0, int(min_mixed_clients))
    if out[_ROLE_MIXED] >= min_mixed:
        return out

    need = int(min_mixed - out[_ROLE_MIXED])
    out[_ROLE_MIXED] += need
    for donor in sorted([_ROLE_BENIGN, _ROLE_ATTACK], key=lambda k: out[k], reverse=True):
        if need <= 0:
            break
        take = min(need, out[donor])
        out[donor] -= take
        need -= take
    if need > 0:
        raise ValueError("partition.params.min_mixed_clients is too large for n_clients/per_bin.")
    return out


def _select_sources_source_stratified(
    source_stats: dict[str, dict[str, float | int]],
    n_clients: int,
    low_thr: float,
    high_thr: float,
    mixed_center: float,
    mixed_width: float,
    min_mixed_clients: int,
    fallback_strategy: str,
    per_bin_raw: list[int] | None,
    seed: int,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    if not (0.0 <= low_thr <= high_thr <= 1.0):
        raise ValueError("partition.params low/high thresholds must satisfy 0 <= low_thr <= high_thr <= 1.")
    if not (0.0 <= mixed_center <= 1.0):
        raise ValueError("partition.params.mixed_center must be in [0,1].")
    if mixed_width < 0.0:
        raise ValueError("partition.params.mixed_width must be >= 0.")

    targets = _normalize_targets(n_clients=n_clients, per_bin_raw=per_bin_raw)
    targets = _enforce_min_mixed_target(targets=targets, min_mixed_clients=min_mixed_clients)
    rng = np.random.default_rng(seed)
    tie_break = {src: float(rng.random()) for src in source_stats.keys()}
    ratios = {src: float(stats["attack_ratio"]) for src, stats in source_stats.items()}
    n_train = {src: int(stats["n_train"]) for src, stats in source_stats.items()}
    if len(ratios) < int(n_clients):
        raise ValueError(
            f"Could not select requested n_clients={n_clients}; only {len(ratios)} eligible sources."
        )

    mixed_lo = max(0.0, float(mixed_center - mixed_width))
    mixed_hi = min(1.0, float(mixed_center + mixed_width))
    bin_key_to_role = {"benign": _ROLE_BENIGN, "mixed": _ROLE_MIXED, "attack": _ROLE_ATTACK}
    role_by_source = {
        src: _role_from_ratio(attack_ratio=ratios[src], low_thr=low_thr, high_thr=high_thr)
        for src in ratios.keys()
    }

    def _rank(srcs: list[str], key_fn):
        return sorted(srcs, key=lambda s: (key_fn(s), -n_train[s], tie_break[s], s))

    selected_by_bin: dict[str, list[str]] = {"benign": [], "mixed": [], "attack": []}
    selected_set: set[str] = set()

    def _take_into(bin_name: str, candidates: list[str], need: int) -> int:
        take = candidates[: max(0, int(need))]
        added = 0
        for src in take:
            if src in selected_set:
                continue
            selected_by_bin[bin_name].append(src)
            selected_set.add(src)
            added += 1
        return max(0, int(need) - int(added))

    # 1) benign bin: lowest attack ratio from benign pool.
    benign_pool = [src for src, role in role_by_source.items() if role == _ROLE_BENIGN]
    benign_ranked = _rank(benign_pool, key_fn=lambda s: ratios[s])
    need_benign = int(targets[_ROLE_BENIGN])
    need_benign = _take_into("benign", benign_ranked, need_benign)

    # 2) attack bin: highest attack ratio from attack pool.
    attack_pool = [src for src, role in role_by_source.items() if role == _ROLE_ATTACK]
    attack_ranked = _rank(attack_pool, key_fn=lambda s: -ratios[s])
    need_attack = int(targets[_ROLE_ATTACK])
    need_attack = _take_into("attack", attack_ranked, need_attack)

    # 3) mixed bin: prefer center window, then fallback to closest-to-center if needed.
    mixed_need = int(targets[_ROLE_MIXED])
    remaining = [src for src in ratios.keys() if src not in selected_set]
    mixed_in_range = [src for src in remaining if mixed_lo <= ratios[src] <= mixed_hi]
    mixed_ranked = _rank(mixed_in_range, key_fn=lambda s: abs(ratios[s] - mixed_center))
    mixed_need = _take_into("mixed", mixed_ranked, mixed_need)

    if mixed_need > 0:
        fallback_mode = str(fallback_strategy).strip().lower()
        if fallback_mode == "closest":
            remaining = [src for src in ratios.keys() if src not in selected_set]
            mixed_fallback = _rank(remaining, key_fn=lambda s: abs(ratios[s] - mixed_center))
            mixed_need = _take_into("mixed", mixed_fallback, mixed_need)
        elif fallback_mode in {"none", "disabled"}:
            pass
        else:
            raise ValueError("partition.params.fallback_strategy must be one of: closest, none")

    # Backfill any remaining benign/attack shortages from remaining sources.
    if need_benign > 0:
        remaining = [src for src in ratios.keys() if src not in selected_set]
        benign_fallback = _rank(remaining, key_fn=lambda s: ratios[s])
        need_benign = _take_into("benign", benign_fallback, need_benign)
    if need_attack > 0:
        remaining = [src for src in ratios.keys() if src not in selected_set]
        attack_fallback = _rank(remaining, key_fn=lambda s: -ratios[s])
        need_attack = _take_into("attack", attack_fallback, need_attack)

    # Final fill to n_clients (rare), still deterministic and center-oriented.
    selected_total = sum(len(v) for v in selected_by_bin.values())
    if selected_total < int(n_clients):
        remaining = [src for src in ratios.keys() if src not in selected_set]
        final_ranked = _rank(remaining, key_fn=lambda s: abs(ratios[s] - mixed_center))
        _take_into("mixed", final_ranked, int(n_clients) - selected_total)

    selected_sources = selected_by_bin["benign"] + selected_by_bin["mixed"] + selected_by_bin["attack"]
    selected_sources = selected_sources[: int(n_clients)]
    if len(selected_sources) < int(n_clients):
        raise ValueError(
            f"Could not select requested n_clients={n_clients}; only {len(selected_sources)} were selected."
        )

    selected_roles = {
        src: _role_from_ratio(attack_ratio=ratios[src], low_thr=low_thr, high_thr=high_thr) for src in selected_sources
    }
    selection_debug: dict[str, Any] = {
        "selected_src_ips_by_bin": {
            "benign": list(selected_by_bin["benign"]),
            "mixed": list(selected_by_bin["mixed"]),
            "attack": list(selected_by_bin["attack"]),
        },
        "bin_counts_available": {
            "benign": int(sum(1 for src in ratios.keys() if role_by_source[src] == _ROLE_BENIGN)),
            "mixed": int(sum(1 for src in ratios.keys() if role_by_source[src] == _ROLE_MIXED)),
            "attack": int(sum(1 for src in ratios.keys() if role_by_source[src] == _ROLE_ATTACK)),
        },
        "bin_counts_selected": {
            "benign": int(len(selected_by_bin["benign"])),
            "mixed": int(len(selected_by_bin["mixed"])),
            "attack": int(len(selected_by_bin["attack"])),
        },
        "attack_ratio_per_src_ip_selected": {str(src): float(ratios[src]) for src in selected_sources},
        "thresholds_used": {
            "low_thr": float(low_thr),
            "high_thr": float(high_thr),
            "mixed_center": float(mixed_center),
            "mixed_width": float(mixed_width),
        },
        "effective_targets": {
            "benign": int(targets[bin_key_to_role["benign"]]),
            "mixed": int(targets[bin_key_to_role["mixed"]]),
            "attack": int(targets[bin_key_to_role["attack"]]),
        },
    }
    return selected_sources, selected_roles, selection_debug


def build_real_source_data(
    dataset_name: str,
    cfg: Any,
    partition_cfg: Any | None = None,
) -> SyntheticDataBundle:
    seed = int(_cfg_get(cfg, "seed", 0))
    df = _load_df(cfg)
    n_rows_loaded = int(df.shape[0])
    if n_rows_loaded <= 0:
        raise ValueError("Loaded empty dataframe from data_path.")

    label_col = _resolve_col(
        df,
        explicit=_cfg_get(cfg, "label_col", None),
        candidates=list(_cfg_get(cfg, "label_col_candidates", []) or []),
        kind="label",
    )
    source_col = _resolve_col(
        df,
        explicit=_cfg_get(cfg, "source_col", None),
        candidates=list(_cfg_get(cfg, "source_col_candidates", []) or []),
        kind="source_id",
    )

    df = df.loc[df[label_col].notna() & df[source_col].notna()].copy()
    if df.empty:
        raise ValueError("No rows remain after dropping missing label/source_id.")

    benign_labels = list(_cfg_get(cfg, "benign_labels", []) or [])
    attack_labels = list(_cfg_get(cfg, "attack_labels", []) or [])
    y_all = _to_binary_labels(df[label_col], benign_labels=benign_labels, attack_labels=attack_labels)
    source_all = df[source_col].astype(str).to_numpy()

    drop_cols = list(_cfg_get(cfg, "drop_cols", []) or [])
    x_df = _prepare_features(df, label_col=label_col, source_col=source_col, drop_cols=drop_cols)

    test_frac = float(_cfg_get(cfg, "global_test_frac", 0.2))
    val_frac = float(_cfg_get(cfg, "global_val_frac", 0.1))
    (
        x_train_df,
        y_train_all,
        src_train,
        x_val_df,
        y_val_all,
        src_val,
        x_test_df,
        y_test,
    ) = _split_global(
        x_df,
        y_all,
        source_all,
        test_frac=test_frac,
        val_frac=val_frac,
        seed=seed,
    )

    fillna_mode = str(_cfg_get(cfg, "fillna", "median"))
    x_train_df, x_val_df, x_test_df = _fillna_with_train_stats(
        x_train=x_train_df,
        x_val=x_val_df,
        x_test=x_test_df,
        mode=fillna_mode,
    )

    x_train_np = x_train_df.to_numpy(dtype=np.float32, copy=False)
    x_val_np = x_val_df.to_numpy(dtype=np.float32, copy=False)
    x_test_np = x_test_df.to_numpy(dtype=np.float32, copy=False)
    standardize_mode = str(_cfg_get(cfg, "standardize_mode", "global_train_only"))
    x_train_np, x_val_np, x_test_np, norm_debug = _standardize_global_train_only(
        x_train=x_train_np,
        x_val=x_val_np,
        x_test=x_test_np,
        mode=standardize_mode,
    )

    part_name = str(_cfg_get(partition_cfg, "name", "none")).strip().lower() if partition_cfg is not None else "none"
    part_params = _cfg_get(partition_cfg, "params", {}) if partition_cfg is not None else {}
    n_clients_raw = _cfg_get(part_params, "max_clients", None)
    if n_clients_raw is None:
        n_clients_raw = _cfg_get(part_params, "n_clients", None)
    requested_n_clients = None if n_clients_raw is None else int(n_clients_raw)
    min_train_samples = int(
        _cfg_get(part_params, "min_train_samples", _cfg_get(part_params, "min_samples_per_client", 1))
    )
    min_val_samples = int(_cfg_get(part_params, "min_val_samples", 0))
    part_seed = int(_cfg_get(part_params, "seed", seed))
    role_low_thr = float(_cfg_get(part_params, "low_thr", 0.05))
    role_high_thr = float(_cfg_get(part_params, "high_thr", 0.95))
    mixed_center = float(_cfg_get(part_params, "mixed_center", 0.5))
    mixed_width = float(_cfg_get(part_params, "mixed_width", 0.3))
    min_mixed_clients = int(_cfg_get(part_params, "min_mixed_clients", 3))
    fallback_strategy = str(_cfg_get(part_params, "fallback_strategy", "closest"))
    per_bin_raw = _cfg_get(part_params, "per_bin", None)
    if per_bin_raw is not None:
        per_bin_raw = list(per_bin_raw)
    if min_train_samples <= 0:
        raise ValueError("partition.params.min_train_samples must be > 0")
    if min_val_samples < 0:
        raise ValueError("partition.params.min_val_samples must be >= 0")
    if not (0.0 <= role_low_thr <= role_high_thr <= 1.0):
        raise ValueError("partition thresholds must satisfy 0 <= low_thr <= high_thr <= 1")
    if not (0.0 <= mixed_center <= 1.0):
        raise ValueError("partition.params.mixed_center must be in [0,1]")
    if mixed_width < 0.0:
        raise ValueError("partition.params.mixed_width must be >= 0")
    if min_mixed_clients < 0:
        raise ValueError("partition.params.min_mixed_clients must be >= 0")

    attack_ratio_threshold = float(_cfg_get(cfg, "attack_ratio_threshold", 0.8))
    if not (0.0 <= attack_ratio_threshold <= 1.0):
        raise ValueError("attack_ratio_threshold must be in [0,1]")

    src_train_arr = np.asarray(src_train)
    src_val_arr = np.asarray(src_val)
    source_stats_all = _build_source_stats(src_train_arr=src_train_arr, y_train_all=y_train_all, src_val_arr=src_val_arr)

    source_stats: dict[str, dict[str, float | int]] = {}
    dropped_small = 0
    for src, stats in source_stats_all.items():
        if int(stats["n_train"]) < min_train_samples:
            dropped_small += 1
            continue
        if int(stats["n_val"]) < min_val_samples:
            dropped_small += 1
            continue
        source_stats[src] = stats

    if not source_stats:
        raise ValueError(
            "No eligible sources after min_samples filtering. Relax partition constraints."
        )

    source_roles_all = {
        src: _role_from_ratio(float(stats["attack_ratio"]), role_low_thr, role_high_thr) for src, stats in source_stats.items()
    }
    bin_counts_available = {
        "benign": int(sum(1 for src in source_stats.keys() if source_roles_all[src] == _ROLE_BENIGN)),
        "mixed": int(sum(1 for src in source_stats.keys() if source_roles_all[src] == _ROLE_MIXED)),
        "attack": int(sum(1 for src in source_stats.keys() if source_roles_all[src] == _ROLE_ATTACK)),
    }

    if part_name == "source_stratified":
        if requested_n_clients is None:
            requested_n_clients = 10
        selected_sources, selected_roles, strat_debug = _select_sources_source_stratified(
            source_stats=source_stats,
            n_clients=int(requested_n_clients),
            low_thr=role_low_thr,
            high_thr=role_high_thr,
            mixed_center=mixed_center,
            mixed_width=mixed_width,
            min_mixed_clients=min_mixed_clients,
            fallback_strategy=fallback_strategy,
            per_bin_raw=per_bin_raw,
            seed=part_seed,
        )
        selected_src_ips_by_bin = dict(strat_debug.get("selected_src_ips_by_bin", {}))
        bin_counts_available = dict(strat_debug.get("bin_counts_available", bin_counts_available))
        bin_counts_selected = dict(strat_debug.get("bin_counts_selected", {}))
        attack_ratio_per_src_ip_selected = dict(strat_debug.get("attack_ratio_per_src_ip_selected", {}))
        thresholds_used = dict(
            strat_debug.get(
                "thresholds_used",
                {
                    "low_thr": float(role_low_thr),
                    "high_thr": float(role_high_thr),
                    "mixed_center": float(mixed_center),
                    "mixed_width": float(mixed_width),
                },
            )
        )
        partition_effective_targets = dict(strat_debug.get("effective_targets", {}))
    else:
        # source_k / none: deterministic top-k by train size.
        ordered = sorted(source_stats.keys(), key=lambda s: (-int(source_stats[s]["n_train"]), str(s)))
        if requested_n_clients is None:
            selected_sources = ordered
        else:
            selected_sources = ordered[: max(0, int(requested_n_clients))]
        selected_roles = {
            src: _role_from_ratio(float(source_stats[src]["attack_ratio"]), role_low_thr, role_high_thr)
            for src in selected_sources
        }
        selected_src_ips_by_bin = {
            "benign": [str(src) for src in selected_sources if selected_roles[src] == _ROLE_BENIGN],
            "mixed": [str(src) for src in selected_sources if selected_roles[src] == _ROLE_MIXED],
            "attack": [str(src) for src in selected_sources if selected_roles[src] == _ROLE_ATTACK],
        }
        bin_counts_selected = {
            "benign": int(len(selected_src_ips_by_bin["benign"])),
            "mixed": int(len(selected_src_ips_by_bin["mixed"])),
            "attack": int(len(selected_src_ips_by_bin["attack"])),
        }
        attack_ratio_per_src_ip_selected = {str(src): float(source_stats[src]["attack_ratio"]) for src in selected_sources}
        thresholds_used = {
            "low_thr": float(role_low_thr),
            "high_thr": float(role_high_thr),
            "mixed_center": float(mixed_center),
            "mixed_width": float(mixed_width),
        }
        partition_effective_targets = {}

    if not selected_sources:
        raise ValueError(
            "Source partitioning produced zero clients. Relax partition params or verify source column values."
        )

    source_to_client: dict[str, int] = {}
    clients: list[ClientDataset] = []
    source_id_per_client: list[str] = []
    attack_ratio_per_client_list: list[float] = []
    client_role_labels: list[str] = []

    for src in selected_sources:
        tr_idx = np.where(src_train_arr == src)[0]
        va_idx = np.where(src_val_arr == src)[0]

        cid = len(clients)
        source_to_client[str(src)] = cid
        x_tr = x_train_np[tr_idx]
        y_tr = y_train_all[tr_idx].astype(np.int64, copy=False)
        x_va = x_val_np[va_idx] if va_idx.size > 0 else np.zeros((0, x_train_np.shape[1]), dtype=np.float32)
        y_va = y_val_all[va_idx].astype(np.int64, copy=False) if va_idx.size > 0 else np.zeros((0,), dtype=np.int64)

        attack_ratio = float(np.mean(y_tr == 1)) if y_tr.size > 0 else 0.0
        is_attacker = bool(attack_ratio >= attack_ratio_threshold)
        clients.append(
            ClientDataset(
                client_id=cid,
                x_train=x_tr.astype(np.float32, copy=False),
                y_train=y_tr,
                x_val=x_va.astype(np.float32, copy=False),
                y_val=y_va,
                is_noisy=False,
                is_attacker=is_attacker,
            )
        )
        source_id_per_client.append(str(src))
        attack_ratio_per_client_list.append(attack_ratio)
        client_role_labels.append(selected_roles[str(src)])

    assigned_val_mask = np.isin(src_val_arr, np.asarray(source_id_per_client, dtype=object))
    n_unassigned_val_rows = int((~assigned_val_mask).sum())
    attacker_client_ids = [int(c.client_id) for c in clients if bool(c.is_attacker)]
    attack_ratio_per_client = {
        str(src): float(ratio) for src, ratio in zip(source_id_per_client, attack_ratio_per_client_list)
    }
    bin_counts = {
        _ROLE_BENIGN: int(sum(1 for src in selected_sources if selected_roles[src] == _ROLE_BENIGN)),
        _ROLE_MIXED: int(sum(1 for src in selected_sources if selected_roles[src] == _ROLE_MIXED)),
        _ROLE_ATTACK: int(sum(1 for src in selected_sources if selected_roles[src] == _ROLE_ATTACK)),
    }

    debug = {
        "dataset_name": str(dataset_name),
        "label_col": str(label_col),
        "source_col": str(source_col),
        "n_rows_loaded": int(n_rows_loaded),
        "n_rows_used": int(df.shape[0]),
        "n_feature_columns": int(x_train_np.shape[1]),
        "n_global_train": int(y_train_all.size),
        "n_global_val": int(y_val_all.size),
        "n_test": int(y_test.size),
        "global_train_pos_rate": float(np.mean(y_train_all == 1)),
        "global_val_pos_rate": float(np.mean(y_val_all == 1)),
        "test_pos_rate": float(np.mean(y_test == 1)),
        "n_clients": int(len(clients)),
        "selected_client_ids": source_id_per_client,
        "source_id_per_client": source_id_per_client,
        "attack_ratio_threshold": float(attack_ratio_threshold),
        "attack_ratio_per_client": attack_ratio_per_client,
        "attack_ratio_per_client_list": attack_ratio_per_client_list,
        "client_role_labels": client_role_labels,
        "bin_counts": bin_counts,
        "selected_src_ips_by_bin": selected_src_ips_by_bin,
        "bin_counts_available": bin_counts_available,
        "bin_counts_selected": bin_counts_selected,
        "attack_ratio_per_src_ip_selected": attack_ratio_per_src_ip_selected,
        "low_thr": float(role_low_thr),
        "high_thr": float(role_high_thr),
        "mixed_center": float(mixed_center),
        "mixed_width": float(mixed_width),
        "thresholds_used": thresholds_used,
        "attacker_client_ids": attacker_client_ids,
        "num_attacker_clients": int(len(attacker_client_ids)),
        "n_noisy_clients": 0,
        "n_attack_clients": int(len(attacker_client_ids)),
        "noise_client_ids": [],
        "attack_client_ids": attacker_client_ids,
        "partition_name": part_name,
        "partition_max_clients": requested_n_clients,
        "partition_min_train_samples": int(min_train_samples),
        "partition_min_val_samples": int(min_val_samples),
        "partition_per_bin": per_bin_raw,
        "partition_effective_targets": partition_effective_targets,
        "partition_min_mixed_clients": int(min_mixed_clients),
        "partition_fallback_strategy": str(fallback_strategy),
        "partition_sources_dropped_small": int(dropped_small),
        "n_unassigned_val_rows": int(n_unassigned_val_rows),
        "n_client_train_labels_flipped_noisy": 0,
        "n_client_train_labels_flipped_attack": 0,
        "n_client_val_labels_flipped": 0,
        "n_global_val_labels_flipped": 0,
        "n_global_test_labels_flipped": 0,
        "label_flip_scope": "none",
        "global_val_label_flip_applied": False,
        "global_test_label_flip_applied": False,
        "attack_mode": "none",
        "attack_scale": 1.0,
    }
    debug.update(norm_debug)

    return SyntheticDataBundle(
        clients=clients,
        x_global_val=x_val_np.astype(np.float32, copy=False),
        y_global_val=y_val_all.astype(np.int64, copy=False),
        x_test=x_test_np.astype(np.float32, copy=False),
        y_test=y_test.astype(np.int64, copy=False),
        debug=debug,
    )
