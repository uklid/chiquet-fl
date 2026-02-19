from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROLE_ORDER = ["good", "noisy", "attacker"]


def _normalize_role(x: str) -> str:
    role = str(x).strip().lower()
    if role in {"clean", "good"}:
        return "good"
    if role in {"noisy"}:
        return "noisy"
    if role in {"attacker", "attack"}:
        return "attacker"
    return "good"


def _parse_run_dirs(value: str | None) -> list[str]:
    if value is None or value.strip() == "":
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _safe_load_results(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _discover_run_dirs(inputs: list[str]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    for raw in inputs:
        p = Path(raw)
        if p.is_file():
            if p.name == "client_scores_round.csv":
                run_dir = str(p.parent.resolve())
                if run_dir not in seen:
                    seen.add(run_dir)
                    found.append(run_dir)
            continue

        if not p.is_dir():
            continue

        direct = p / "client_scores_round.csv"
        if direct.exists():
            run_dir = str(p.resolve())
            if run_dir not in seen:
                seen.add(run_dir)
                found.append(run_dir)

        for csv_path in p.rglob("client_scores_round.csv"):
            run_dir = str(csv_path.parent.resolve())
            if run_dir not in seen:
                seen.add(run_dir)
                found.append(run_dir)

    return found


def _results_lookup(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if df.empty or "run_dir" not in df.columns:
        return lookup

    for _, row in df.iterrows():
        run_dir_raw = row.get("run_dir")
        if pd.isna(run_dir_raw):
            continue
        run_dir_path = Path(str(run_dir_raw))
        keys = {str(run_dir_path), str((Path.cwd() / run_dir_path).resolve())}
        payload = {
            "dataset": row.get("dataset"),
            "scorer": row.get("scorer"),
            "aggregator": row.get("aggregator"),
        }
        for key in keys:
            lookup[key] = payload

    return lookup


def _read_cfg_meta(run_dir: str) -> dict[str, Any]:
    cfg_path = Path(run_dir) / "config_resolved.yaml"
    if not cfg_path.exists():
        return {}
    try:
        cfg = OmegaConf.load(cfg_path)
        cfg_dict = OmegaConf.to_container(cfg, resolve=False)
    except Exception:
        return {}
    if not isinstance(cfg_dict, dict):
        return {}

    dataset = None
    scorer = None
    aggregator = None
    if isinstance(cfg_dict.get("dataset"), dict):
        dataset = cfg_dict["dataset"].get("name")
    if isinstance(cfg_dict.get("scorer"), dict):
        scorer = cfg_dict["scorer"].get("name")
    if isinstance(cfg_dict.get("aggregator"), dict):
        aggregator = cfg_dict["aggregator"].get("name")

    return {"dataset": dataset, "scorer": scorer, "aggregator": aggregator}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_master.csv")
    ap.add_argument("--scorer", default="cosine")
    ap.add_argument("--run_dirs", default="")
    ap.add_argument("--out_png", default="paper_assets/synth_scores.png")
    ap.add_argument("--out_pdf", default="paper_assets/synth_scores.pdf")
    args = ap.parse_args()

    # Run enumeration never requires results_master: if --run_dirs is empty,
    # discover runs recursively under ./runs by default.
    raw_inputs = _parse_run_dirs(args.run_dirs)
    if raw_inputs:
        run_dirs = _discover_run_dirs(raw_inputs)
    else:
        run_dirs = _discover_run_dirs(["runs"])
    if not run_dirs:
        raise ValueError("No run directories found. Provide --run_dirs or ensure runs/*/* exists.")

    res = _safe_load_results(args.results)
    res_lookup = _results_lookup(res)

    rows = []
    used_runs = 0
    desired_scorer = str(args.scorer).strip().lower()
    for run_dir in run_dirs:
        # Prefer metadata from results_master if present; fall back to run config.
        meta = {}
        if run_dir in res_lookup:
            meta = res_lookup[run_dir]
        else:
            meta = _read_cfg_meta(run_dir)

        run_scorer = str(meta.get("scorer", "")).strip().lower()
        if desired_scorer and run_scorer and run_scorer != desired_scorer:
            continue

        score_path = Path(run_dir) / "client_scores_round.csv"
        if not score_path.exists():
            continue
        score_df = pd.read_csv(score_path)
        if score_df.empty or "round" not in score_df.columns or "score" not in score_df.columns:
            continue
        if "client_role" not in score_df.columns:
            continue

        final_round = int(pd.to_numeric(score_df["round"], errors="coerce").max())
        sub = score_df[pd.to_numeric(score_df["round"], errors="coerce") == final_round].copy()
        if sub.empty:
            continue
        sub["score"] = pd.to_numeric(sub["score"], errors="coerce")
        sub = sub[sub["score"].notna()]
        if sub.empty:
            continue

        sub["client_role"] = sub["client_role"].map(_normalize_role)
        sub["run_dir"] = run_dir
        rows.append(sub[["run_dir", "client_role", "score"]])
        used_runs += 1

    if not rows:
        raise ValueError("No usable client score files found for selected runs.")

    plot_df = pd.concat(rows, ignore_index=True)
    data = [plot_df.loc[plot_df["client_role"] == role, "score"].to_numpy(dtype=float) for role in ROLE_ORDER]

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 4.8))
    positions = np.arange(1, len(ROLE_ORDER) + 1)
    bp_data = [vals if vals.size > 0 else np.array([np.nan]) for vals in data]
    plt.boxplot(
        bp_data,
        positions=positions,
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#dceefe", "edgecolor": "#4472c4"},
        medianprops={"color": "#1f1f1f"},
    )

    rng = np.random.default_rng(0)
    for idx, vals in enumerate(data, start=1):
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=vals.size)
        plt.scatter(np.full(vals.size, idx) + jitter, vals, s=18, alpha=0.7, color="#1f77b4")

    labels = [f"{role}\n(n={int(len(vals))})" for role, vals in zip(ROLE_ORDER, data)]
    plt.xticks(positions, labels)
    plt.ylim(0.0, 1.02)
    plt.ylabel("Reliability Score (Final Round)")
    plt.title(f"Client Score Distribution by Role ({args.scorer}, {used_runs} run(s))")
    plt.grid(axis="y", alpha=0.25, linewidth=0.6)
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
