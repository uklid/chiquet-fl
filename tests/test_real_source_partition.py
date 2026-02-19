from __future__ import annotations

import numpy as np
import pandas as pd

from fl.datasets.real_source import build_real_source_data


def test_real_source_partition_and_attacker_metadata(tmp_path) -> None:
    rng = np.random.default_rng(7)
    n_per_source = 120

    src = np.array(["host_a"] * n_per_source + ["host_b"] * n_per_source + ["host_c"] * n_per_source, dtype=object)
    y_a = np.array(["Attack"] * n_per_source, dtype=object)
    y_b = np.array(["Benign"] * n_per_source, dtype=object)
    y_c = np.where(rng.random(n_per_source) < 0.5, "Attack", "Benign").astype(object)
    y = np.concatenate([y_a, y_b, y_c], axis=0)

    f1 = rng.normal(size=3 * n_per_source)
    f2 = rng.normal(size=3 * n_per_source)
    df = pd.DataFrame({"source_id": src, "label": y, "f1": f1, "f2": f2})
    csv_path = tmp_path / "toy_real.csv"
    df.to_csv(csv_path, index=False)

    data = build_real_source_data(
        dataset_name="toy_real",
        cfg={
            "data_path": str(csv_path),
            "label_col": "label",
            "source_col": "source_id",
            "drop_cols": [],
            "benign_labels": ["Benign"],
            "attack_labels": None,
            "global_test_frac": 0.2,
            "global_val_frac": 0.1,
            "standardize_mode": "global_train_only",
            "fillna": "median",
            "attack_ratio_threshold": 0.8,
            "seed": 7,
        },
        partition_cfg={
            "name": "source_k",
            "params": {"max_clients": None, "min_train_samples": 10, "min_val_samples": 0},
        },
    )

    assert len(data.clients) >= 3
    debug = data.debug
    assert debug["attack_ratio_threshold"] == 0.8
    assert debug["num_attacker_clients"] >= 1
    assert debug["n_client_train_labels_flipped_attack"] == 0
    assert debug["label_flip_scope"] == "none"
    assert debug["standardize_mode"] == "global_train_only"
    assert debug["scaler_fit_scope"] == "global_train"
    assert debug["n_global_val_used_for_scaler"] == 0
    assert debug["n_global_test_used_for_scaler"] == 0

    # host_a is all-attack and should be marked attacker after source-based partition.
    source_ids = debug["source_id_per_client"]
    attack_ratios = debug["attack_ratio_per_client"]
    if isinstance(attack_ratios, dict):
        source_to_ratio = {str(k): float(v) for k, v in attack_ratios.items()}
    else:
        source_to_ratio = {sid: float(r) for sid, r in zip(source_ids, attack_ratios)}
    assert source_to_ratio["host_a"] > 0.95


def test_real_source_source_stratified_backfill_and_metadata(tmp_path) -> None:
    rng = np.random.default_rng(11)
    n_per_source = 80

    # No mixed source here on purpose, so per_bin mixed quota requires backfill.
    sources = (
        ["src_b0"] * n_per_source
        + ["src_b1"] * n_per_source
        + ["src_a0"] * n_per_source
        + ["src_a1"] * n_per_source
    )
    y_b0 = np.array(["Benign"] * n_per_source, dtype=object)
    y_b1 = np.array(["Benign"] * n_per_source, dtype=object)
    y_a0 = np.array(["Attack"] * n_per_source, dtype=object)
    y_a1 = np.array(["Attack"] * n_per_source, dtype=object)
    y = np.concatenate([y_b0, y_b1, y_a0, y_a1], axis=0)

    f1 = rng.normal(size=4 * n_per_source)
    f2 = rng.normal(size=4 * n_per_source)
    df = pd.DataFrame({"src_ip": np.asarray(sources, dtype=object), "label": y, "f1": f1, "f2": f2})
    csv_path = tmp_path / "toy_real_stratified.csv"
    df.to_csv(csv_path, index=False)

    data = build_real_source_data(
        dataset_name="toy_real",
        cfg={
            "data_path": str(csv_path),
            "label_col": "label",
            "source_col": "src_ip",
            "drop_cols": [],
            "benign_labels": ["Benign"],
            "attack_labels": None,
            "global_test_frac": 0.2,
            "global_val_frac": 0.1,
            "standardize_mode": "global_train_only",
            "fillna": "median",
            "attack_ratio_threshold": 0.8,
            "seed": 11,
        },
        partition_cfg={
            "name": "source_stratified",
            "params": {
                "n_clients": 4,
                "min_samples_per_client": 10,
                "min_val_samples": 0,
                "low_thr": 0.05,
                "high_thr": 0.95,
                "per_bin": [1, 2, 1],
                "seed": 11,
            },
        },
    )

    debug = data.debug
    assert debug["partition_name"] == "source_stratified"
    assert len(debug["selected_client_ids"]) == 4
    assert isinstance(debug["attack_ratio_per_client"], dict)
    assert set(debug["attack_ratio_per_client"].keys()) == set(debug["selected_client_ids"])
    assert int(sum(int(v) for v in debug["bin_counts"].values())) == 4
