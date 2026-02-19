from __future__ import annotations

import numpy as np

from .base import Aggregator, AggregationContext


class MedianAggregator(Aggregator):
    def aggregate(self, client_updates: np.ndarray, weights: np.ndarray, ctx: AggregationContext) -> np.ndarray:
        del weights
        if client_updates.ndim != 2:
            raise ValueError(f"Expected client_updates shape [n_clients, d], got {client_updates.shape}")
        n_clients = int(client_updates.shape[0])
        if n_clients == 0:
            raise ValueError("Expected at least one client update.")
        d = int(client_updates.shape[1])
        weight_i = np.zeros((n_clients,), dtype=np.float64)
        if d > 0:
            half = n_clients // 2
            for j in range(d):
                order = np.argsort(client_updates[:, j], kind="mergesort")
                if n_clients % 2 == 1:
                    weight_i[int(order[half])] += 1.0
                else:
                    weight_i[int(order[half - 1])] += 0.5
                    weight_i[int(order[half])] += 0.5
            weight_i = weight_i / float(d)
        else:
            weight_i[:] = 1.0 / n_clients
        ctx.meta["weight_i_used"] = weight_i
        ctx.meta["weights_sum_to_one"] = True
        ctx.meta["weight_mode"] = "coordinate_influence_mean"
        return np.median(client_updates, axis=0).astype(np.float64, copy=False)
