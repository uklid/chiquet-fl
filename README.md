# Choquet-FL Reliability

## Hydra Multirun Ablations

### Ablation B
Fix `scorer=cosine`, vary aggregator, run 3 seeds.

```bash
python -m fl.run -m experiment=synth_ablation_B aggregator=fedavg,median,trimmed_mean,choquet_additive,choquet_2add seed=0,1,2
```

### FedAvg Baselines
Run pure FedAvg (`fedavg|none`) and cosine-weighted FedAvg (`fedavg|cosine`) with 3 seeds each.

```bash
python -m fl.run -m experiment=synth_fedavg_pure seed=0,1,2
python -m fl.run -m experiment=synth_fedavg_cosine seed=0,1,2
```

## Collate + Assets

```bash
python scripts/collate_results.py --runs_dir runs --out results_master.csv
python scripts/make_plots_synth.py --results results_master.csv --out_png paper_assets/synth_robustness.png --out_pdf paper_assets/synth_robustness.pdf
python scripts/make_tables.py --results results_master.csv --out_tex paper_assets/synth_table.tex
```
