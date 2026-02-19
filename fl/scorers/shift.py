from __future__ import annotations

import numpy as np

from .base import ClientReliabilityScorer, ScoringContext


class LabelShiftScorer(ClientReliabilityScorer):
    def __init__(self, k: float = 4.0, metric: str = "js", eps: float = 1e-12):
        self.k = float(k)
        self.metric = str(metric).lower()
        self.eps = float(eps)
        if self.metric not in {"js", "l1"}:
            raise ValueError(f"Unknown shift metric '{metric}'. Use 'js' or 'l1'.")

    def _hist_binary(self, y: np.ndarray) -> np.ndarray:
        y_int = np.asarray(y, dtype=np.int64)
        counts = np.bincount(y_int, minlength=2).astype(np.float64)
        return counts / (counts.sum() + self.eps)

    def _js_div(self, p: np.ndarray, q: np.ndarray) -> float:
        m = 0.5 * (p + q)
        p_safe = np.clip(p, self.eps, 1.0)
        q_safe = np.clip(q, self.eps, 1.0)
        m_safe = np.clip(m, self.eps, 1.0)
        kl_pm = float(np.sum(p_safe * np.log(p_safe / m_safe)))
        kl_qm = float(np.sum(q_safe * np.log(q_safe / m_safe)))
        js = 0.5 * (kl_pm + kl_qm)
        return float(js / np.log(2.0))

    def _divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        if self.metric == "js":
            return self._js_div(p, q)
        # 0.5 * L1 keeps range in [0, 1].
        return float(0.5 * np.sum(np.abs(p - q)))

    def score_clients(self, client_updates: np.ndarray, ctx: ScoringContext) -> np.ndarray:
        del client_updates
        clients = ctx.meta.get("clients")
        global_hist = ctx.meta.get("global_label_hist")
        if clients is None:
            raise ValueError("LabelShiftScorer requires ctx.meta['clients'].")

        if global_hist is None:
            y_all = np.concatenate([np.asarray(client.y_train, dtype=np.int64) for client in clients], axis=0)
            global_hist = self._hist_binary(y_all)
        else:
            global_hist = np.asarray(global_hist, dtype=np.float64)

        scores = []
        for client in clients:
            y_train = np.asarray(client.y_train, dtype=np.int64)
            if y_train.size == 0:
                scores.append(0.5)
                continue
            client_hist = self._hist_binary(y_train)
            div = self._divergence(client_hist, global_hist)
            rel = float(np.exp(-self.k * div))
            scores.append(rel)

        return np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
