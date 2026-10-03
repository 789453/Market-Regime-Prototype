import numpy as np

from src.crypto.predictive_states_v2 import (center_membership, covariance_geometry,
                                             fit_region_probabilities, region_predict)
from scripts.crypto_predictive_v2_middle_path import middle_summaries


def test_soft_regions_are_normalized_and_smooth_near_a_boundary():
    center = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    x = np.array([[.99, 0], [1.01, 0], [0, .1]], dtype=np.float32)
    hard, distance, soft, margin = center_membership(x, center, .5)
    assert np.allclose(soft.sum(axis=1), 1, atol=1e-6)
    assert hard[0] != hard[1]
    assert abs(float(soft[0, 0] - soft[1, 0])) < .05
    assert distance[2] < distance[0]
    assert np.all(margin >= 0)


def test_weighted_region_probabilities_preserve_all_task_simplexes():
    y = np.array([[0, 1, 0], [0, 2, 1], [4, 4, 1], [4, 3, 0]], dtype=np.int8)
    symbol = np.array([0, 0, 1, 1], dtype=np.int8)
    weight = np.array([[.9, .1], [.7, .3], [.2, .8], [.1, .9]], dtype=np.float32)
    general, local = fit_region_probabilities(y, symbol, weight, alpha=2, symbol_alpha=2)
    p = region_predict(weight, symbol, general, local)
    assert [item.shape for item in p] == [(4, 5), (4, 5), (4, 2)]
    assert all(np.allclose(item.sum(axis=1), 1, atol=1e-6) for item in p)
    assert p[0][0, 0] > p[0][2, 0]


def test_covariance_geometry_limits_duplicate_axis_and_uses_training_only():
    rng = np.random.default_rng(8)
    base = rng.normal(size=200)
    x = np.column_stack((base, base + rng.normal(scale=.01, size=200),
                         rng.normal(size=200), rng.normal(size=200))).astype(np.float32)
    first, maps = covariance_geometry(x, np.arange(100), group_width=4)
    changed = x.copy()
    changed[100:] += 20
    second, maps_after = covariance_geometry(changed, np.arange(100), group_width=4)
    assert np.allclose(maps[0], maps_after[0])
    assert np.allclose(first[:100], second[:100])
    assert np.isfinite(first).all()


def test_middle_path_uses_ordered_past_nodes():
    x = np.zeros((2, 100), dtype=np.float32)
    # The first economic group stores: current[0:4], 3h[6:8],
    # 2h[8:10] and 1h[10:12]. Same current/endpoints, different middle.
    x[:, 2:4] = 4
    x[0, 6:8], x[0, 8:10], x[0, 10:12] = 0, 1, 3
    x[1, 6:8], x[1, 8:10], x[1, 10:12] = 3, 1, 0
    summary, names = middle_summaries(x)
    assert summary.shape == (2, 18)
    assert names[0] == "vol_strength_change3h"
    assert summary[0, 0] != summary[1, 0]
    assert summary[0, 1] == summary[1, 1]
    assert summary[0, 2] != summary[1, 2]
