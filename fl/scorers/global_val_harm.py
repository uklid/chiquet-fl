from __future__ import annotations

import numpy as np

from .base import ClientReliabilityScorer, ScoringContext


class GlobalValHarmScorer(ClientReliabilityScorer):
    name = "global_val_harm"
    score_semantics = "harm"

    def __init__(
        self,
        beta: float = 10.0,
        clamp_positive: bool = True,
        subset_size: int | None = None,
        seed: int = 0,
        eps: float = 1e-12,
    ):
        self.beta = float(beta)
        self.clamp_positive = bool(clamp_positive)
        self.subset_size = None if subset_size is None else int(subset_size)
        self.seed = int(seed)
        self.eps = float(eps)
        if self.beta < 0.0:
            raise ValueError("beta must be >= 0.0.")
        if self.subset_size is not None and self.subset_size <= 0:
            raise ValueError("subset_size must be > 0 when provided.")
        if self.eps <= 0.0:
            raise ValueError("eps must be > 0.")

    def _subset(
        self,
        x: np.ndarray,
        y: np.ndarray,
        round_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = int(y.shape[0])
        if self.subset_size is None or self.subset_size >= n:
            return x, y
        rng = np.random.default_rng(self.seed + int(round_idx))
        idx = rng.choice(n, size=self.subset_size, replace=False)
        return x[idx], y[idx]

    def score_clients(self, client_updates: np.ndarray, ctx: ScoringContext) -> np.ndarray:
        if client_updates.ndim != 2:
            raise ValueError(f"Expected 2D client_updates [n_clients, d], got shape {client_updates.shape}")

        n_clients = int(client_updates.shape[0])
        if n_clients == 0:
            return np.zeros((0,), dtype=np.float64)

        global_params = ctx.meta.get("global_params")
        x_global_val = ctx.meta.get("x_global_val")
        y_global_val = ctx.meta.get("y_global_val")
        eval_loss_at_params = ctx.meta.get("eval_loss_at_params")
        if global_params is None or x_global_val is None or y_global_val is None or not callable(eval_loss_at_params):
            raise ValueError(
                "GlobalValHarmScorer requires ctx.meta['global_params'], "
                "ctx.meta['x_global_val'], ctx.meta['y_global_val'], and callable "
                "ctx.meta['eval_loss_at_params']."
            )

        params0 = np.asarray(global_params, dtype=np.float64).reshape(-1)
        x_val = np.asarray(x_global_val, dtype=np.float32)
        y_val = np.asarray(y_global_val, dtype=np.int64).reshape(-1)
        if y_val.size == 0:
            return np.zeros((n_clients,), dtype=np.float64)

        x_sub, y_sub = self._subset(x_val, y_val, ctx.round_idx)
        loss_before = float(eval_loss_at_params(params0, x_sub, y_sub))

        loss_after = np.zeros((n_clients,), dtype=np.float64)
        delta = np.zeros((n_clients,), dtype=np.float64)
        for i in range(n_clients):
            params_after = params0 + np.asarray(client_updates[i], dtype=np.float64)
            li = float(eval_loss_at_params(params_after, x_sub, y_sub))
            loss_after[i] = li
            delta[i] = li - loss_before

        if self.clamp_positive:
            harm = np.maximum(delta, 0.0)
        else:
            harm = delta.copy()
        harm_for_reliability = np.maximum(harm, 0.0)
        reliability = np.exp(-self.beta * harm_for_reliability)
        reliability = np.clip(reliability, 0.0, 1.0).astype(np.float64, copy=False)

        if not np.isfinite(loss_before):
            raise ValueError("Non-finite global validation loss before client updates.")
        if not np.all(np.isfinite(loss_after)):
            raise ValueError("Non-finite global validation loss after client updates.")
        if not np.all(np.isfinite(delta)):
            raise ValueError("Non-finite global validation delta loss values.")
        if not np.all(np.isfinite(harm)):
            raise ValueError("Non-finite harm scores.")
        if not np.all(np.isfinite(reliability)):
            raise ValueError("Non-finite reliability values.")

        ctx.meta["score_semantics"] = "harm"
        ctx.meta["global_val_loss_before"] = float(loss_before)
        ctx.meta["global_val_loss_after_i"] = loss_after
        ctx.meta["global_val_delta_loss_i"] = delta
        ctx.meta["harm_score_i"] = harm
        ctx.meta["reliability_i"] = reliability
        return harm.astype(np.float64, copy=False)
