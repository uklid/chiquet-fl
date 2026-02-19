from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class ScoringContext:
    round_idx: int
    meta: Dict[str, Any]

class ClientReliabilityScorer:
    def fit_round(self, client_updates: np.ndarray, ctx: ScoringContext) -> None:
        return

    def score_clients(self, client_updates: np.ndarray, ctx: ScoringContext) -> np.ndarray:
        raise NotImplementedError

    def pairwise_scores(self, client_updates: np.ndarray, ctx: ScoringContext) -> Optional[np.ndarray]:
        return None
