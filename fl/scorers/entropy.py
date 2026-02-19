from __future__ import annotations

import numpy as np

from .base import ClientReliabilityScorer, ScoringContext


class EntropyScorer(ClientReliabilityScorer):
    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)
        self._max_entropy = float(np.log(2.0))

    def score_clients(self, client_updates: np.ndarray, ctx: ScoringContext) -> np.ndarray:
        del client_updates
        clients = ctx.meta.get("clients")
        global_params = ctx.meta.get("global_params")
        predict_proba_fn = ctx.meta.get("global_predict_proba")
        if clients is None or global_params is None:
            raise ValueError("EntropyScorer requires ctx.meta['clients'] and ctx.meta['global_params'].")

        params = np.asarray(global_params, dtype=np.float64)
        scores = []

        for client in clients:
            x_val = np.asarray(client.x_val, dtype=np.float64)
            if x_val.size == 0:
                scores.append(0.5)
                continue

            if callable(predict_proba_fn):
                probs = np.asarray(predict_proba_fn(x_val), dtype=np.float64)
                if probs.ndim != 2 or probs.shape[0] != x_val.shape[0] or probs.shape[1] != 2:
                    raise ValueError(
                        "ctx.meta['global_predict_proba'] must return array with shape [n_samples, 2]."
                    )
                probs = np.clip(probs, self.eps, 1.0)
            else:
                if params.shape != (3,):
                    raise ValueError(
                        "EntropyScorer expected global_params shape (3,) for linear model when "
                        "ctx.meta['global_predict_proba'] is not provided."
                    )
                w = params[:2]
                b = float(params[2])
                logits = x_val @ w + b
                logits = np.clip(logits, -60.0, 60.0)
                p1 = 1.0 / (1.0 + np.exp(-logits))
                probs = np.stack([1.0 - p1, p1], axis=1)
                probs = np.clip(probs, self.eps, 1.0)

            ent = -np.sum(probs * np.log(probs), axis=1)
            mean_ent = float(np.mean(ent))
            rel = 1.0 - (mean_ent / self._max_entropy)
            scores.append(rel)

        return np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
