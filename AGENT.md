# Codex Agent: Choquet-FL + Client Reliability (NO Sugeno)

## Mission
Build a reproducible federated learning research codebase to compare Choquet integral aggregation vs baselines
with a primary focus on multiple client reliability scoring methods.

Strict constraint: DO NOT implement Sugeno integral.

## Non-negotiables
- Config-driven experiments (Hydra YAML).
- Deterministic seeds (>=3 seeds/setting).
- Standard artifacts per run:
  - metrics_round.csv
  - summary.json
  - config_resolved.yaml
  - client_scores_round.csv (when scorers used)
- Collation script to produce results_master.csv from runs/.

## Core abstractions (must be plugins)
- ClientReliabilityScorer: score_clients(...) -> scores in [0,1]; optional pairwise_scores(...) -> matrix
- Aggregator: aggregate(client_updates, scores/weights, context) -> global_update
- FL Engine: server rounds + client local training (dataset later)

## Choquet modes
- Additive Choquet (must match weighted average)
- 2-additive Choquet with interaction terms (use pairwise compatibility score initially)

## Two-way ablation requirement
A) Fix aggregator=Choquet; vary scorer (R1..)
B) Fix scorer (e.g., R1); vary aggregator (FedAvg, robust, Choquet)

## Acceptance checklist
- python -m fl.run runs and writes artifacts under runs/<exp>/<timestamp>/
- sanity test passes: additive Choquet == weighted average within tolerance
- scripts/collate_results.py outputs results_master.csv
