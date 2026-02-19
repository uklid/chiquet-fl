from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _method_label(row: pd.Series) -> str:
    agg = str(row["aggregator"])
    sc = str(row["scorer"])
    return f"{agg}|{sc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_master.csv")
    ap.add_argument("--out_png", default="paper_assets/synth_robustness.png")
    ap.add_argument("--out_pdf", default="paper_assets/synth_robustness.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    df = df.copy()

    if "final_test_acc" not in df.columns:
        raise ValueError("results CSV must include final_test_acc column.")
    if "noise_rate" not in df.columns:
        raise ValueError("results CSV must include noise_rate column.")

    df["final_test_acc"] = pd.to_numeric(df["final_test_acc"], errors="coerce")
    df["noise_rate"] = pd.to_numeric(df["noise_rate"], errors="coerce")
    df["noise_clients_frac"] = pd.to_numeric(df.get("noise_clients_frac"), errors="coerce")
    df["attack_clients_frac"] = pd.to_numeric(df.get("attack_clients_frac"), errors="coerce")
    df["seed"] = pd.to_numeric(df.get("seed"), errors="coerce")

    df = df[
        (df["dataset"] == "synth2d")
        & df["final_test_acc"].notna()
        & df["noise_rate"].notna()
        & ~df["run_dir"].astype(str).str.contains("smoke", case=False, na=False)
    ].copy()
    if "final_metric_name" in df.columns:
        keep_metric = df["final_metric_name"].isna() | (df["final_metric_name"].astype(str) == "final_test_acc")
        df = df[keep_metric].copy()
    if df.empty:
        raise ValueError("No synth2d rows with final_test_acc/noise_rate found.")

    df["method"] = df.apply(_method_label, axis=1)

    group_cols = [
        "dataset",
        "noise_rate",
        "noise_clients_frac",
        "attack_clients_frac",
        "aggregator",
        "scorer",
    ]
    per_seed = (
        df.groupby(group_cols + ["seed"], dropna=False)["final_test_acc"]
        .mean()
        .reset_index()
    )
    g = (
        per_seed.groupby(group_cols, dropna=False)["final_test_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["aggregator", "scorer", "noise_rate", "noise_clients_frac", "attack_clients_frac"])
    )
    g["method"] = g["aggregator"].astype(str) + "|" + g["scorer"].astype(str)

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    methods = sorted(g["method"].unique().tolist())
    for method in methods:
        sub_all = g[g["method"] == method].copy()
        scenarios = (
            sub_all[["noise_clients_frac", "attack_clients_frac"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        for nc, atk in scenarios:
            sub = sub_all[
                (sub_all["noise_clients_frac"] == nc) & (sub_all["attack_clients_frac"] == atk)
            ].sort_values("noise_rate")
            x = sub["noise_rate"].to_numpy(dtype=float)
            y = sub["mean"].to_numpy(dtype=float)
            s = sub["std"].fillna(0.0).to_numpy(dtype=float)
            label = method if len(sub_all[["noise_clients_frac", "attack_clients_frac"]].drop_duplicates()) == 1 else (
                f"{method} [nc={nc:.2f}, atk={atk:.2f}]"
            )
            plt.plot(x, y, marker="o", linewidth=1.8, label=label)
            if np.any(s > 0):
                plt.fill_between(x, y - s, y + s, alpha=0.15)

    plt.xlabel("Noise Rate")
    plt.ylabel("Final Test Accuracy")
    plt.title("Synthetic Robustness: Final Accuracy vs Noise Rate")
    plt.grid(alpha=0.25, linewidth=0.6)
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

    out_pdf = Path(args.out_pdf)
    if out_pdf.suffix.lower() == ".pdf":
        plt.savefig(out_pdf)
    plt.close()

    print(f"[OK] Wrote {out_png}")
    if out_pdf.suffix.lower() == ".pdf":
        print(f"[OK] Wrote {out_pdf}")


if __name__ == "__main__":
    main()
