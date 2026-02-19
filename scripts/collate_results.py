from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import argparse
from omegaconf import OmegaConf


def _nested_get(d: dict, keys: list[str], default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _pick_final_metric(summary: dict) -> tuple[object, str]:
    # Normalize final metric into final_test_acc column for downstream scripts.
    # If no accuracy metric exists, fall back to available terminal metric.
    for key in ("final_test_acc", "final_val_acc", "final_acc"):
        if key in summary:
            return summary.get(key), key
    for key in ("final_stub_metric", "final_loss", "final_val_loss"):
        if key in summary:
            return summary.get(key), key
    return None, ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out", default="results_master.csv")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    rows = []

    for summary_path in runs_dir.glob("*/*/summary.json"):
        run_dir = summary_path.parent
        cfg_path = run_dir / "config_resolved.yaml"
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        cfg_dict = {}
        if cfg_path.exists():
            cfg = OmegaConf.load(cfg_path)
            cfg_dict = OmegaConf.to_container(cfg, resolve=False)

        dataset_name = _nested_get(cfg_dict, ["dataset", "name"])
        aggregator_name = _nested_get(cfg_dict, ["aggregator", "name"])
        scorer_name = _nested_get(cfg_dict, ["scorer", "name"])
        seed = _nested_get(cfg_dict, ["seed"])
        noise_rate = _nested_get(cfg_dict, ["dataset", "params", "noise_rate"])
        noise_clients_frac = _nested_get(cfg_dict, ["dataset", "params", "noise_clients_frac"])
        attack_clients_frac = _nested_get(cfg_dict, ["dataset", "params", "attack_clients_frac"])
        final_test_acc = summary.get("final_test_acc")
        final_metric_name = summary.get("final_metric_name", "")
        if final_test_acc is None:
            final_test_acc, fallback_name = _pick_final_metric(summary)
            if not final_metric_name:
                final_metric_name = fallback_name

        row = {
            "run_dir": str(run_dir),
            "dataset": dataset_name,
            "aggregator": aggregator_name,
            "scorer": scorer_name,
            "seed": seed,
            "noise_rate": noise_rate,
            "noise_clients_frac": noise_clients_frac,
            "attack_clients_frac": attack_clients_frac,
            "final_test_acc": final_test_acc,
            "final_metric_name": final_metric_name,
            "runtime_sec": summary.get("runtime_sec"),
            "has_config": cfg_path.exists(),
        }
        rows.append(row)

    cols = [
        "run_dir",
        "dataset",
        "aggregator",
        "scorer",
        "seed",
        "noise_rate",
        "noise_clients_frac",
        "attack_clients_frac",
        "final_test_acc",
        "final_metric_name",
        "runtime_sec",
        "has_config",
    ]
    df = pd.DataFrame(rows, columns=cols).sort_values("run_dir") if rows else pd.DataFrame(columns=cols)
    df.to_csv(args.out, index=False)
    print(f"[OK] Wrote {args.out} with {len(df)} rows")

if __name__ == "__main__":
    main()
