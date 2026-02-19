from __future__ import annotations
import numpy as np
from .base import ClientReliabilityScorer, ScoringContext

class NoScorer(ClientReliabilityScorer):
    def score_clients(self, client_updates: np.ndarray, ctx: ScoringContext) -> np.ndarray:
        # Uniform reliability
        return np.ones((client_updates.shape[0],), dtype=np.float64)
