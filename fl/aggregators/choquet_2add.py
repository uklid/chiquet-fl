from __future__ import annotations

import numpy as np

from .base import Aggregator, AggregationContext


class Choquet2AddAggregator(Aggregator):
    """
    Fast 2-additive Choquet-style aggregation.

    Uses reliability singletons as base capacity and pairwise interactions from
    cosine similarities between client updates.
    """

    def __init__(self, interaction_strength: float = 0.2, positive_only: bool = True, eps: float = 1e-12):
        self.interaction_strength = float(interaction_strength)
        self.positive_only = bool(positive_only)
        self.eps = float(eps)

    def aggregate(self, client_updates: np.ndarray, weights: np.ndarray, ctx: AggregationContext) -> np.ndarray:
        if client_updates.ndim != 2:
            raise ValueError(f"Expected client_updates shape [n_clients, d], got {client_updates.shape}")
        n_clients = int(client_updates.shape[0])
        if n_clients == 0:
            raise ValueError("Expected at least one client update.")

        base = np.asarray(weights, dtype=np.float64).reshape(-1)
        if base.shape[0] != n_clients:
            raise ValueError(f"Weights length mismatch: got {base.shape[0]}, expected {n_clients}.")

        base = np.clip(base, 0.0, None)
        base_sum = float(base.sum())
        if base_sum <= self.eps:
            base = np.full((n_clients,), 1.0 / n_clients, dtype=np.float64)
        else:
            base = base / base_sum

        if n_clients == 1:
            ctx.meta["weight_i_used"] = np.array([1.0], dtype=np.float64)
            ctx.meta["weights_sum_to_one"] = True
            ctx.meta["weight_mode"] = "choquet_2add_effective"
            return client_updates[0].astype(np.float64, copy=False)

        norms = np.linalg.norm(client_updates, axis=1)
        normalized = client_updates / (norms[:, None] + self.eps)
        cosine = normalized @ normalized.T
        np.fill_diagonal(cosine, 0.0)

        if self.positive_only:
            similarity = np.clip(cosine, 0.0, 1.0)
        else:
            similarity = 0.5 * (cosine + 1.0)
            similarity = np.clip(similarity, 0.0, 1.0)
            np.fill_diagonal(similarity, 0.0)

        pair_scale = np.sqrt(np.outer(base, base))
        b = self.interaction_strength * similarity * pair_scale
        b = 0.5 * (b + b.T)
        np.fill_diagonal(b, 0.0)

        # Split each pairwise term equally across both clients to derive
        # a stable effective weight vector in O(n^2 + nd).
        effective = base + 0.5 * b.sum(axis=1)
        effective = np.clip(effective, 0.0, None)
        eff_sum = float(effective.sum())
        if eff_sum <= self.eps:
            effective = base
        else:
            effective = effective / eff_sum

        ctx.meta["weight_i_used"] = effective.copy()
        ctx.meta["weights_sum_to_one"] = True
        ctx.meta["weight_mode"] = "choquet_2add_effective"
        return (client_updates * effective[:, None]).sum(axis=0)
