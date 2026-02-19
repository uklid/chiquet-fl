from __future__ import annotations

import os
from typing import Any, Mapping
from pathlib import Path
import pandas as pd
import hydra
from omegaconf import OmegaConf

from fl.utils.io import ensure_dir, save_json
from fl.utils.seeding import seed_all
from fl.registry import build_aggregator, build_scorer
from fl.engine import run_stub_engine, run_synth2d_engine
from fl.datasets.synthetic import build_synthetic_data
from fl.datasets.real_source import build_real_source_data


def _with_input_dim(model_cfg: Any, input_dim: int) -> dict[str, Any]:
    if not isinstance(model_cfg, Mapping):
        return {"name": "logreg", "params": {"input_dim": int(input_dim)}}

    out = dict(model_cfg)
    name = str(out.get("name", "logreg")).strip()
    if name.lower() in {"", "none"}:
        name = "logreg"
    params_raw = out.get("params", {})
    params = dict(params_raw) if isinstance(params_raw, Mapping) else {}
    if name.lower() in {"logreg", "linear_logreg_2d", "mlp_small"}:
        params["input_dim"] = int(input_dim)
    out["name"] = name
    out["params"] = params
    return out


def _client_stats_df(clients, dataset_debug: Mapping[str, Any] | None = None) -> pd.DataFrame:
    rows = []
    role_labels = None
    if isinstance(dataset_debug, Mapping):
        maybe_roles = dataset_debug.get("client_role_labels")
        if isinstance(maybe_roles, list) and len(maybe_roles) == len(clients):
            role_labels = [str(r) for r in maybe_roles]
    for client in clients:
        cid = int(client.client_id)
        if role_labels is not None and 0 <= cid < len(role_labels):
            role = role_labels[cid]
        elif client.is_attacker:
            role = "attacker"
        elif client.is_noisy:
            role = "noisy"
        else:
            role = "clean"
        rows.append(
            {
                "client_id": cid,
                "role": role,
                "n_train": int(client.y_train.size),
                "pos_rate_train": float(client.y_train.mean()) if client.y_train.size > 0 else 0.0,
                "n_val": int(client.y_val.size),
                "pos_rate_val": float(client.y_val.mean()) if client.y_val.size > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("client_id").reset_index(drop=True)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    seed_all(int(cfg.seed))

    out_root = Path(cfg.run.out_dir) / str(cfg.run.exp_name) / str(cfg.run.timestamp)
    ensure_dir(out_root)

    # Save resolved config
    if cfg.run.save_resolved_config:
        (out_root / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg))

    aggregator = build_aggregator(OmegaConf.to_container(cfg.aggregator, resolve=True))
    scorer = build_scorer(OmegaConf.to_container(cfg.scorer, resolve=True))
    model_node = OmegaConf.select(cfg, "model", default=None)
    model_cfg = OmegaConf.to_container(model_node, resolve=True) if model_node is not None else None
    partition_node = OmegaConf.select(cfg, "partition", default=None)
    partition_cfg = (
        OmegaConf.to_container(partition_node, resolve=True)
        if partition_node is not None
        else {"name": "none", "params": {}}
    )
    client_data_stats_df = None

    dataset_name = str(cfg.dataset.name)
    if dataset_name == "none":
        outputs = run_stub_engine(
            rounds=int(cfg.fl.rounds),
            n_clients=int(os.environ.get("N_CLIENTS", "10")),
            dim=int(os.environ.get("UPDATE_DIM", "50")),
            aggregator=aggregator,
            scorer=scorer,
        )
    elif dataset_name == "synth2d":
        dataset_params = OmegaConf.to_container(cfg.dataset.params, resolve=True)
        dataset_params["seed"] = int(cfg.seed)
        data = build_synthetic_data(dataset_params)
        client_data_stats_df = _client_stats_df(data.clients, dataset_debug=data.debug)
        resolved_model_cfg = _with_input_dim(model_cfg, int(data.x_global_val.shape[1]))
        outputs = run_synth2d_engine(
            clients=data.clients,
            x_global_val=data.x_global_val,
            y_global_val=data.y_global_val,
            x_test=data.x_test,
            y_test=data.y_test,
            dataset_debug=data.debug,
            model_cfg=resolved_model_cfg,
            rounds=int(cfg.fl.rounds),
            sample_fraction=float(cfg.fl.sample_fraction),
            local_epochs=int(cfg.fl.local_epochs),
            batch_size=int(cfg.fl.batch_size),
            lr=float(cfg.fl.lr),
            fl_method=str(cfg.fl.method),
            fedprox_mu=float(cfg.fl.fedprox_mu),
            optimizer_name=str(cfg.fl.optimizer),
            aggregator=aggregator,
            scorer=scorer,
            device_name=str(cfg.device),
            seed=int(cfg.seed),
            debug_fedprox=bool(cfg.fl.debug_fedprox),
            attack_mode=str(dataset_params.get("attack_mode", "label_flip")),
            attack_scale=float(dataset_params.get("attack_scale", 1.0)),
        )
    elif dataset_name in {"cicids2017", "ton_iot", "ctu13"}:
        dataset_params = OmegaConf.to_container(cfg.dataset.params, resolve=True)
        dataset_params["seed"] = int(cfg.seed)
        data = build_real_source_data(dataset_name=dataset_name, cfg=dataset_params, partition_cfg=partition_cfg)
        client_data_stats_df = _client_stats_df(data.clients, dataset_debug=data.debug)
        resolved_model_cfg = _with_input_dim(model_cfg, int(data.x_global_val.shape[1]))
        outputs = run_synth2d_engine(
            clients=data.clients,
            x_global_val=data.x_global_val,
            y_global_val=data.y_global_val,
            x_test=data.x_test,
            y_test=data.y_test,
            dataset_debug=data.debug,
            model_cfg=resolved_model_cfg,
            rounds=int(cfg.fl.rounds),
            sample_fraction=float(cfg.fl.sample_fraction),
            local_epochs=int(cfg.fl.local_epochs),
            batch_size=int(cfg.fl.batch_size),
            lr=float(cfg.fl.lr),
            fl_method=str(cfg.fl.method),
            fedprox_mu=float(cfg.fl.fedprox_mu),
            optimizer_name=str(cfg.fl.optimizer),
            aggregator=aggregator,
            scorer=scorer,
            device_name=str(cfg.device),
            seed=int(cfg.seed),
            debug_fedprox=bool(cfg.fl.debug_fedprox),
            # Real-dataset mode: attacker metadata only, no update poisoning.
            attack_mode="label_flip",
            attack_scale=1.0,
        )
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")

    outputs.metrics_round.to_csv(out_root / "metrics_round.csv", index=False)
    outputs.client_scores_round.to_csv(out_root / "client_scores_round.csv", index=False)
    if outputs.fedprox_debug_round is not None:
        outputs.fedprox_debug_round.to_csv(out_root / "fedprox_debug_round.csv", index=False)
    if outputs.fedprox_debug is not None:
        outputs.fedprox_debug.to_csv(out_root / "fedprox_debug.csv", index=False)
    if client_data_stats_df is not None:
        client_data_stats_df.to_csv(out_root / "client_data_stats.csv", index=False)
    save_json(out_root / "summary.json", outputs.summary)

    print(f"[OK] Wrote run artifacts to: {out_root}")

if __name__ == "__main__":
    main()
