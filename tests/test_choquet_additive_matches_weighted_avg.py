from __future__ import annotations

import numpy as np

from fl.aggregators.base import AggregationContext
from fl.aggregators.choquet_additive import ChoquetAdditiveAggregator
from fl.aggregators.fedavg import FedAvgAggregator


def test_choquet_additive_matches_weighted_avg() -> None:
    rng = np.random.default_rng(12345)
    ctx = AggregationContext(round_idx=0, meta={})
    fedavg = FedAvgAggregator()
    choquet = ChoquetAdditiveAggregator()

    for _ in range(20):
        n_clients = int(rng.integers(3, 32))
        dim = int(rng.integers(2, 128))
        updates = rng.normal(loc=0.0, scale=1.0, size=(n_clients, dim))
        weights = rng.uniform(low=1e-3, high=2.0, size=(n_clients,))

        ref = fedavg.aggregate(updates, weights, ctx)
        got = choquet.aggregate(updates, weights, ctx)
        np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-12)
