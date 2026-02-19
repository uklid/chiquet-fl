from __future__ import annotations

import numpy as np

from .base import ClientReliabilityScorer, ScoringContext


class CosineScorer(ClientReliabilityScorer):
    def __init__(
        self,
        reference: str = "median",
        trim_ratio: float = 0.1,
        eps: float = 1e-12,
    ):
        self.reference = str(reference).lower()
        self.trim_ratio = float(trim_ratio)
        self.eps = float(eps)
        if self.reference not in {"median", "trimmed_mean", "mean"}:
            raise ValueError("reference must be one of: median, trimmed_mean, mean")
        if not (0.0 <= self.trim_ratio < 0.5):
            raise ValueError("trim_ratio must be in [0.0, 0.5).")

    def _reference_direction(self, client_updates: np.ndarray) -> np.ndarray:
        if self.reference == "median":
            return np.median(client_updates, axis=0).astype(np.float64, copy=False)
        if self.reference == "mean":
            return np.mean(client_updates, axis=0).astype(np.float64, copy=False)

        # trimmed_mean: coordinate-wise trimmed mean
        n_clients = int(client_updates.shape[0])
        k = int(np.floor(self.trim_ratio * n_clients))
        if k <= 0:
            return np.mean(client_updates, axis=0).astype(np.float64, copy=False)
        if 2 * k >= n_clients:
            return np.median(client_updates, axis=0).astype(np.float64, copy=False)
        sorted_updates = np.sort(client_updates, axis=0)
        trimmed = sorted_updates[k : n_clients - k, :]
        return np.mean(trimmed, axis=0).astype(np.float64, copy=False)

    def score_clients(self, client_updates: np.ndarray, ctx: ScoringContext) -> np.ndarray:
        if client_updates.ndim != 2:
            raise ValueError(f"Expected 2D client_updates [n_clients, d], got shape {client_updates.shape}")

        n_clients = int(client_updates.shape[0])
        if n_clients == 0:
            return np.zeros((0,), dtype=np.float64)

        ref = self._reference_direction(client_updates)
        ref_norm = float(np.linalg.norm(ref))
        if ref_norm <= self.eps:
            cos = np.zeros((n_clients,), dtype=np.float64)
            weights = np.full((n_clients,), 1.0 / n_clients, dtype=np.float64)
            ctx.meta["cos_i"] = cos
            ctx.meta["weight_i"] = weights
            ctx.meta["scorer_weights"] = weights
            ctx.meta["cosine_reference"] = self.reference
            return weights

        client_norms = np.linalg.norm(client_updates, axis=1)
        numer = client_updates @ ref
        denom = client_norms * ref_norm
        cos = np.divide(numer, denom + self.eps)
        cos = np.where(client_norms <= self.eps, 0.0, cos)

        hinge = np.clip(cos, 0.0, None)
        hinge_sum = float(np.sum(hinge))
        if hinge_sum <= self.eps:
            weights = np.full((n_clients,), 1.0 / n_clients, dtype=np.float64)
        else:
            weights = hinge / hinge_sum

        weights = np.clip(weights, 0.0, 1.0).astype(np.float64, copy=False)
        ctx.meta["cos_i"] = cos.astype(np.float64, copy=False)
        ctx.meta["weight_i"] = weights
        ctx.meta["scorer_weights"] = weights
        ctx.meta["cosine_reference"] = self.reference
        return weights
