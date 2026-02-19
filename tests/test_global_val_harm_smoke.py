from __future__ import annotations

import numpy as np

from fl.aggregators.fedavg import FedAvgAggregator
from fl.datasets.synthetic import build_synthetic_data
from fl.engine import run_synth2d_engine
from fl.scorers.global_val_harm import GlobalValHarmScorer


def test_global_val_harm_smoke_attackers_have_higher_harm_and_weights_sum_to_one() -> None:
    seed = 2
    data = build_synthetic_data(
        {
            "num_clients": 8,
            "n_samples_total": 640,
            "quantity_skew": 0.0,
            "label_skew": 0.0,
            "noise_clients_frac": 0.0,
            "noise_rate": 0.0,
            "attack_clients_frac": 0.25,
            "class_sep": 1.5,
            "flip_y": 0.0,
            "label_noise_rate": 0.0,
            "val_frac": 0.2,
            "n_global_val_samples": 256,
            "n_test_samples": 256,
            "standardize_mode": "global_train_only",
            "seed": seed,
        }
    )

    outputs = run_synth2d_engine(
        clients=data.clients,
        x_global_val=data.x_global_val,
        y_global_val=data.y_global_val,
        x_test=data.x_test,
        y_test=data.y_test,
        dataset_debug=data.debug,
        model_cfg={"name": "logreg"},
        rounds=1,
        sample_fraction=1.0,
        local_epochs=2,
        batch_size=64,
        lr=0.1,
        fl_method="fedavg",
        fedprox_mu=0.0,
        optimizer_name="sgd",
        aggregator=FedAvgAggregator(),
        scorer=GlobalValHarmScorer(beta=10.0, clamp_positive=True, subset_size=None, seed=seed, eps=1e-12),
        device_name="cpu",
        seed=seed,
        debug_fedprox=False,
    )

    scores = outputs.client_scores_round
    required_cols = {
        "round",
        "client_id",
        "client_role",
        "score",
        "harm_score",
        "reliability",
        "global_val_loss_before",
        "global_val_loss_after",
        "global_val_delta_loss",
        "cos_i",
        "weight_i",
    }
    assert required_cols.issubset(set(scores.columns))

    numeric_cols = [
        "harm_score",
        "reliability",
        "global_val_loss_before",
        "global_val_loss_after",
        "global_val_delta_loss",
        "cos_i",
        "weight_i",
    ]
    assert np.isfinite(scores[numeric_cols].to_numpy(dtype=np.float64)).all()

    weight_sums = scores.groupby("round")["weight_i"].sum().to_numpy(dtype=np.float64)
    assert np.allclose(weight_sums, 1.0, atol=1e-6)

    attacker_harm = float(scores.loc[scores["client_role"] == "attacker", "harm_score"].mean())
    clean_harm = float(scores.loc[scores["client_role"] == "clean", "harm_score"].mean())
    assert np.isfinite(attacker_harm)
    assert np.isfinite(clean_harm)
    assert attacker_harm > clean_harm
