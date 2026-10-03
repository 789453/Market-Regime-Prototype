import numpy as np

from src.crypto.regime_prototypes import (
    REGIMES, adaptive_pair_weights, classify_regime, fit_predictive_prototypes,
    future_path_labels, held_pair_score, select_beta_pair, semivariance_direction,
)
from scripts.crypto_calm_down_slow_stage import slow_coordinates


def test_future_side_moments_start_after_executable_open():
    # A jump *into* the entry open is already known at execution.  It must not
    # appear in the future upside or downside statistic.
    changes = np.r_[0.0, .10, np.repeat(.001, 48), np.zeros(10)]
    prices = np.cumsum(changes)
    label = future_path_labels(prices, np.array([1]), horizons=(4,))
    assert np.isclose(label["return4"][0], .048)
    assert np.isclose(label["up4"][0], 48 * .001**2)
    assert np.isclose(label["down4"][0], 0)
    assert np.isclose(semivariance_direction(label["up4"], label["down4"])[0], 1, atol=1e-7)


def test_beta_weighted_candidate_is_actual_held_payoff():
    prediction = np.array([.002, .001, -.001])
    beta = np.array([2.0, .5, 1.0])
    long, short, score = select_beta_pair(prediction, beta)
    assert long != short
    assert np.isclose(score, held_pair_score(prediction, beta, long, short))
    # Highest minus lowest is not generally the largest beta-neutral payoff.
    values = [held_pair_score(prediction, beta, a, b) for a in range(3) for b in range(3) if a != b]
    assert np.isclose(score, max(values))


def test_low_and_high_states_use_only_given_past_thresholds():
    result = classify_regime(np.array([.01, .01, .05, .2]),
                             np.array([-.1, .1, -.1, .1]), .02, .1)
    assert [REGIMES[v] for v in result] == ["calm_down", "calm_up", "normal", "high"]


def test_policy_keeps_then_exits_its_own_pair_even_if_other_pair_is_strong():
    pred = np.array([[.004, -.004, 0., 0.],
                     [.003, -.003, .002, -.002],
                     [-.004, .004, .008, -.008]])
    beta = np.ones_like(pred)
    weights, episodes = adaptive_pair_weights(pred, beta, np.zeros(3, dtype=int),
        np.full(3, .002), np.full(3, .0005), continuous=False, min_hours=3)
    assert weights[0, 0] > 0 and weights[1, 0] > 0
    assert weights[2, 2] > 0 and weights[2, 3] < 0
    assert episodes[2, 2] == 1


def test_predictive_prototype_requires_later_time_gain():
    rng = np.random.default_rng(42)
    n = 5000
    x = rng.normal(size=(n, 4)).astype(np.float32)
    y = np.column_stack((np.where(x[:, 0] > 0, 1., -1.) + rng.normal(0, .1, n),
                         rng.normal(0, .1, n))).astype(np.float32)
    dates = np.repeat(np.arange(100), 50)
    model = fit_predictive_prototypes(x, x[:, :2], y, dates, clusters=1, seed=4)
    assert model.accepted[0]
    forecast, group, leaf = model.predict(x, x[:, :2])
    assert np.mean((forecast[:, 0] - y[:, 0]) ** 2) < np.mean((y[:, 0] - y[:, 0].mean()) ** 2)
    assert np.unique(leaf).size == 2


def test_review_interval_and_switch_hurdle_reduce_churn_without_future_inputs():
    pred = np.array([[.004, -.004, 0., 0.],
                     [.003, -.003, .004, -.004],
                     [.002, -.002, .005, -.005],
                     [.001, -.001, .006, -.006],
                     [.001, -.001, .007, -.007]])
    beta = np.ones_like(pred)
    weights, episodes = adaptive_pair_weights(pred, beta, np.zeros(len(pred), dtype=int),
        np.full(len(pred), .001), np.full(len(pred), .0005), continuous=False,
        review_every_hours=4, switch_hurdle=.0016)
    assert np.array_equal(weights[0], weights[1])
    assert np.array_equal(weights[0], weights[3])
    assert episodes[4, 0] == 2 and episodes[4, 1] == 3


def test_direction_certification_rejects_split_on_risk_only():
    rng = np.random.default_rng(4)
    n = 5000
    x = rng.normal(size=(n, 3)).astype(np.float32)
    y = np.column_stack((rng.normal(0, 1, n),
                         np.where(x[:, 0] > 0, 2., -2.) + rng.normal(0, .1, n))).astype(np.float32)
    dates = np.repeat(np.arange(100), 50)
    risk = fit_predictive_prototypes(x, x[:, :2], y, dates, clusters=1, seed=7)
    direction = fit_predictive_prototypes(x, x[:, :2], y, dates, clusters=1, seed=7,
        fit_task_weights=np.array([1., .1]), certify_tasks=(0,))
    assert risk.accepted[0]
    assert not direction.accepted[0]


def test_slow_features_are_completed_history_only_and_need_30_days():
    rng = np.random.default_rng(12)
    close = rng.normal(0, .002, (900, 2)).cumsum(axis=0)
    rv = np.exp(rng.normal(-7, .1, (900, 2)))
    data = {"close": close, "rv": rv, "beta": np.ones_like(close),
            "market_rv": rv.mean(axis=1)}
    full = slow_coordinates(data)
    prefix = slow_coordinates({key: val[:800] for key, val in data.items()})
    assert np.isnan(full[:720]).all()
    assert np.isfinite(full[720:]).all()
    assert np.allclose(full[720:800], prefix[720:], rtol=0, atol=0)
