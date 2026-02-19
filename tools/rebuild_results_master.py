from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf


def _nested_get(d: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _pick_final_metric(summary: dict[str, Any]) -> tuple[float | None, str]:
    for key in ("final_test_acc", "final_val_acc", "final_acc"):
        if key in summary:
            val = summary.get(key)
            return (None if val is None else float(val), key)
    for key in ("final_test_loss", "final_val_loss", "final_stub_metric", "final_loss"):
        if key in summary:
            val = summary.get(key)
            return (None if val is None else float(val), key)
    return None, ""


def _load_cfg_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=False)  # keep raw interpolation values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out", default="results_master_rebuilt.csv")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    rows: list[dict[str, Any]] = []

    for summary_path in sorted(runs_dir.glob("**/summary.json")):
        run_dir = summary_path.parent
        cfg_path = run_dir / "config_resolved.yaml"
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            continue

        cfg_dict = _load_cfg_dict(cfg_path)
        dataset_params = _nested_get(cfg_dict, ["dataset", "params"], {}) or {}
        partition_params = _nested_get(cfg_dict, ["partition", "params"], {}) or {}

        final_metric = summary.get("final_test_acc")
        final_metric_name = str(summary.get("final_metric_name", ""))
        if final_metric is None:
            final_metric, fallback_name = _pick_final_metric(summary)
            if not final_metric_name:
                final_metric_name = fallback_name

        row = {
            "run_dir": str(run_dir),
            "seed": _nested_get(cfg_dict, ["seed"]),
            "dataset": _nested_get(cfg_dict, ["dataset", "name"]),
            "partition_name": _nested_get(cfg_dict, ["partition", "name"], "none"),
            "partition_max_clients": partition_params.get("max_clients"),
            "partition_min_train_samples": partition_params.get("min_train_samples"),
            "partition_min_val_samples": partition_params.get("min_val_samples"),
            "attack_ratio_threshold": summary.get(
                "attack_ratio_threshold",
                dataset_params.get("attack_ratio_threshold"),
            ),
            "num_attacker_clients": summary.get(
                "num_attacker_clients",
                summary.get("num_attack_clients"),
            ),
            "aggregator": _nested_get(cfg_dict, ["aggregator", "name"]),
            "scorer": _nested_get(cfg_dict, ["scorer", "name"]),
            "fl.method": _nested_get(cfg_dict, ["fl", "method"], summary.get("fl.method")),
            "fedprox_mu": _nested_get(cfg_dict, ["fl", "fedprox_mu"], summary.get("fedprox_mu")),
            "final_metric_name": final_metric_name or "final_test_acc",
            "final_metric": final_metric,
            "final_test_acc": summary.get("final_test_acc"),
            "runtime_sec": summary.get("runtime_sec"),
            "has_config": bool(cfg_path.exists()),
        }
        rows.append(row)

    cols = [
        "run_dir",
        "seed",
        "dataset",
        "partition_name",
        "partition_max_clients",
        "partition_min_train_samples",
        "partition_min_val_samples",
        "attack_ratio_threshold",
        "num_attacker_clients",
        "aggregator",
        "scorer",
        "fl.method",
        "fedprox_mu",
        "final_metric_name",
        "final_metric",
        "final_test_acc",
        "runtime_sec",
        "has_config",
    ]
    df = pd.DataFrame(rows, columns=cols).sort_values("run_dir") if rows else pd.DataFrame(columns=cols)
    df.to_csv(args.out, index=False)
    print(f"[OK] Wrote {args.out} with {len(df)} rows")


if __name__ == "__main__":
    main()
