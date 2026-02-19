from __future__ import annotations
import numpy as np
from .base import Aggregator, AggregationContext

class FedAvgAggregator(Aggregator):
    def aggregate(self, client_updates: np.ndarray, weights: np.ndarray, ctx: AggregationContext) -> np.ndarray:
        if client_updates.ndim != 2:
            raise ValueError(f"Expected client_updates shape [n_clients, d], got {client_updates.shape}")
        n_clients = int(client_updates.shape[0])
        if n_clients == 0:
            raise ValueError("Expected at least one client update.")
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != n_clients:
            raise ValueError(
                f"Weights length mismatch: got {w.shape[0]}, expected {n_clients} for {client_updates.shape}."
            )
        w = np.clip(w, 0.0, None)
        s = float(w.sum())
        if s <= 1e-12:
            w = np.full((n_clients,), 1.0 / n_clients, dtype=np.float64)
        else:
            w = w / s
        ctx.meta["weight_i_used"] = w.copy()
        ctx.meta["weights_sum_to_one"] = True
        ctx.meta["weight_mode"] = "scalar_normalized"
        return (client_updates * w[:, None]).sum(axis=0)
