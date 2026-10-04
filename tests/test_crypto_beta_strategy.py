import numpy as np
import pandas as pd
from src.crypto.beta_strategy import coin_features,markov_filter,adaptive_experts,gate_weights,rebalance_targets

def test_strategy_features_do_not_change_when_future_arrives():
    rng=np.random.default_rng(99);idx=pd.date_range('2024-01-01',periods=1200,freq='h',tz='UTC')
    r=pd.Series(rng.normal(0,.003,len(idx)),index=idx)
    bars=pd.DataFrame(dict(close=np.exp(r.cumsum())*100,quote_volume=100+rng.random(len(idx)),taker_buy_quote_volume=50+rng.random(len(idx))),index=idx)
    full=coin_features(bars,r,pd.Series(.5,index=idx))
    prefix=coin_features(bars.iloc[:1000],r.iloc[:1000],pd.Series(.5,index=idx[:1000]))
    pd.testing.assert_frame_equal(full.iloc[:1000],prefix)

def test_filtered_probabilities_ignore_future_emissions():
    transition=np.array([[.9,.1],[.2,.8]]);initial=np.array([.5,.5])
    emissions=np.array([[.7,.3],[.6,.4],[.1,.9],[.4,.6]])
    a=markov_filter(emissions,transition,initial);b=markov_filter(emissions[:2],transition,initial)
    np.testing.assert_allclose(a[:2],b);np.testing.assert_allclose(a.sum(axis=1),1)

def test_adaptive_action_uses_only_rewards_already_received():
    expert=np.tile(np.linspace(-1,1,7),(30,1));state=np.zeros(30,dtype=np.int64);prior=gate_weights(state)
    rewards=np.zeros_like(expert);rewards[20:,0]=1
    a,wa=adaptive_experts(expert,state,prior,rewards)
    b,wb=adaptive_experts(expert,state,prior,np.zeros_like(rewards))
    np.testing.assert_allclose(a[:20],b[:20]);np.testing.assert_allclose(wa[:20],wb[:20])
    assert a[-1]!=b[-1]

def test_slow_review_keeps_continuous_signals_separate_from_positions():
    desired=np.array([.5,-.5,-.5,-.5,-.5,-.5,-.5,-.5,.1])
    targets,orders=rebalance_targets(desired,np.arange(9),4,.12)
    np.testing.assert_allclose(targets[:4],.5);np.testing.assert_allclose(targets[4:8],-.5)
    assert np.flatnonzero(orders).tolist()==[0,4,8]
