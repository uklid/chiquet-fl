from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _fmt_mean_std(mean: float, std: float, digits: int = 4) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        std = 0.0
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_master.csv")
    ap.add_argument("--out_tex", default="paper_assets/synth_table.tex")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    if "final_test_acc" not in df.columns:
        raise ValueError("results CSV must include final_test_acc.")

    df["final_test_acc"] = pd.to_numeric(df["final_test_acc"], errors="coerce")
    df["runtime_sec"] = pd.to_numeric(df["runtime_sec"], errors="coerce")
    df["noise_rate"] = pd.to_numeric(df.get("noise_rate"), errors="coerce")
    df["noise_clients_frac"] = pd.to_numeric(df.get("noise_clients_frac"), errors="coerce")
    df["attack_clients_frac"] = pd.to_numeric(df.get("attack_clients_frac"), errors="coerce")
    df["seed"] = pd.to_numeric(df.get("seed"), errors="coerce")

    df = df[
        (df["dataset"] == "synth2d")
        & df["final_test_acc"].notna()
        & ~df["run_dir"].astype(str).str.contains("smoke", case=False, na=False)
    ].copy()
    if "final_metric_name" in df.columns:
        keep_metric = df["final_metric_name"].isna() | (df["final_metric_name"].astype(str) == "final_test_acc")
        df = df[keep_metric].copy()
    if df.empty:
        raise ValueError("No synth2d rows with final_test_acc found.")

    group_cols = [
        "dataset",
        "noise_rate",
        "noise_clients_frac",
        "attack_clients_frac",
        "aggregator",
        "scorer",
    ]
    per_seed = (
        df.groupby(group_cols + ["seed"], dropna=False)
        .agg(
            final_test_acc=("final_test_acc", "mean"),
            runtime_sec=("runtime_sec", "mean"),
        )
        .reset_index()
    )
    summary = (
        per_seed.groupby(group_cols, dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            acc_mean=("final_test_acc", "mean"),
            acc_std=("final_test_acc", "std"),
            runtime_mean=("runtime_sec", "mean"),
            runtime_std=("runtime_sec", "std"),
        )
        .reset_index()
        .sort_values(["dataset", "noise_rate", "noise_clients_frac", "attack_clients_frac", "aggregator", "scorer"])
    )
    summary["method"] = summary["aggregator"].astype(str) + "|" + summary["scorer"].astype(str)

    summary["final_test_acc"] = summary.apply(lambda r: _fmt_mean_std(r["acc_mean"], r["acc_std"]), axis=1)
    summary["runtime_sec"] = summary.apply(lambda r: _fmt_mean_std(r["runtime_mean"], r["runtime_std"], 3), axis=1)

    table = summary[
        [
            "dataset",
            "noise_rate",
            "noise_clients_frac",
            "attack_clients_frac",
            "aggregator",
            "scorer",
            "method",
            "n_seeds",
            "final_test_acc",
            "runtime_sec",
        ]
    ].copy()

    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    latex = table.to_latex(index=False, escape=False)
    out_tex.write_text(latex, encoding="utf-8")
    print(f"[OK] Wrote {out_tex}")


if __name__ == "__main__":
    main()
