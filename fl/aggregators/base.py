from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

@dataclass
class AggregationContext:
    round_idx: int
    meta: Dict[str, Any]

class Aggregator:
    def aggregate(self, client_updates: np.ndarray, weights: np.ndarray, ctx: AggregationContext) -> np.ndarray:
        raise NotImplementedError
