from __future__ import annotations

import numpy as np

from .base import Aggregator, AggregationContext


class TrimmedMeanAggregator(Aggregator):
    def __init__(self, trim_ratio: float = 0.1):
        self.trim_ratio = float(trim_ratio)
        if not (0.0 <= self.trim_ratio < 0.5):
            raise ValueError("trim_ratio must be in [0.0, 0.5).")

    def aggregate(self, client_updates: np.ndarray, weights: np.ndarray, ctx: AggregationContext) -> np.ndarray:
        del weights
        if client_updates.ndim != 2:
            raise ValueError(f"Expected client_updates shape [n_clients, d], got {client_updates.shape}")
        n_clients = int(client_updates.shape[0])
        if n_clients == 0:
            raise ValueError("Expected at least one client update.")
        d = int(client_updates.shape[1])
        weight_i = np.zeros((n_clients,), dtype=np.float64)

        k = int(np.floor(self.trim_ratio * n_clients))
        if k <= 0:
            weight_i[:] = 1.0 / n_clients
            ctx.meta["weight_i_used"] = weight_i
            ctx.meta["weights_sum_to_one"] = True
            ctx.meta["weight_mode"] = "uniform_mean"
            return client_updates.mean(axis=0).astype(np.float64, copy=False)
        if 2 * k >= n_clients:
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

        sorted_updates = np.sort(client_updates, axis=0)
        trimmed = sorted_updates[k : n_clients - k, :]
        if d > 0:
            keep_count = n_clients - (2 * k)
            for j in range(d):
                order = np.argsort(client_updates[:, j], kind="mergesort")
                keep = order[k : n_clients - k]
                weight_i[keep] += (1.0 / keep_count)
            weight_i = weight_i / float(d)
        else:
            weight_i[:] = 1.0 / n_clients
        ctx.meta["weight_i_used"] = weight_i
        ctx.meta["weights_sum_to_one"] = True
        ctx.meta["weight_mode"] = "coordinate_influence_mean"
        return trimmed.mean(axis=0).astype(np.float64, copy=False)
