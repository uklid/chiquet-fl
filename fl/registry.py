from __future__ import annotations
from typing import Any, Dict

from fl.aggregators.choquet_2add import Choquet2AddAggregator
from fl.aggregators.choquet_additive import ChoquetAdditiveAggregator
from fl.aggregators.fedavg import FedAvgAggregator
from fl.aggregators.median import MedianAggregator
from fl.aggregators.trimmed_mean import TrimmedMeanAggregator
from fl.scorers.cosine import CosineScorer
from fl.scorers.entropy import EntropyScorer
from fl.scorers.global_val_harm import GlobalValHarmScorer
from fl.scorers.loss_improvement import LossImprovementScorer
from fl.scorers.none import NoScorer
from fl.scorers.shift import LabelShiftScorer

def build_aggregator(cfg: Dict[str, Any]):
    name = cfg.get("name", "fedavg")
    params = cfg.get("params", {}) or {}
    if name == "fedavg":
        return FedAvgAggregator()
    if name == "median":
        return MedianAggregator()
    if name == "trimmed_mean":
        return TrimmedMeanAggregator(**params)
    if name == "choquet_additive":
        return ChoquetAdditiveAggregator()
    if name == "choquet_2add":
        return Choquet2AddAggregator(**params)
    raise ValueError(f"Unknown aggregator: {name}")

def build_scorer(cfg: Dict[str, Any]):
    name = cfg.get("name", "none")
    params = cfg.get("params", {}) or {}
    if name == "none":
        return NoScorer()
    if name == "cosine":
        return CosineScorer(**params)
    if name == "entropy":
        return EntropyScorer(**params)
    if name == "shift":
        return LabelShiftScorer(**params)
    if name == "loss_improvement":
        return LossImprovementScorer(**params)
    if name == "global_val_harm":
        return GlobalValHarmScorer(**params)
    raise ValueError(f"Unknown scorer: {name}")
