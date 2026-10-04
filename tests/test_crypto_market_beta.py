import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.crypto.market_beta import (BOOK_COLUMNS,decision_path,market_features,path_features,
                                   path_labels,quantity_ledger,soft_membership)

CFG=json.loads((Path(__file__).resolve().parents[1]/'configs/crypto_market_beta_a.json').read_text())

def test_no_order_preserves_contract_quantity_and_short_pnl_is_linear():
    price=np.array([[100.,100.],[110.,100.],[121.,100.]])
    book=quantity_ledger(price,np.array([-.5,0.]),np.array([True,False]),np.zeros_like(price),np.array([.5,.5]),0.,False)
    # Short 0.0025 BTC and 0.0025 ETH, no hidden daily rebalance.
    assert np.isclose(book[-1,5],1-.0025*21)
    assert np.isclose(book[1,1]*book[0,5],-.0025*11)
    assert book[1,4]==0

def test_initial_and_final_commissions_solve_post_fee_equity():
    price=np.full((3,2),100.)
    book=quantity_ledger(price,np.array([1.,0.]),np.array([True,False]),np.zeros_like(price),np.array([.5,.5]),10.)
    assert np.isclose(book[0,5],1/1.001)
    assert np.isclose(book[-1,5],(1-.001)/1.001)
    assert np.allclose(book[:,0],book[:,1]-book[:,2]-book[:,3],atol=1e-15)

def test_funding_belongs_to_old_quantity_before_reversal_or_new_entry():
    price=np.full((4,1),100.)
    funding=np.array([[.1],[.02],[.03],[0.]])
    book=quantity_ledger(price,np.array([.5,-.5,0.]),np.array([True,True,False]),funding,np.array([1.]),0.,False)
    assert book[0,3]==0  # newly entered quantity did not own earlier funding.
    assert np.isclose(book[1,3],.5*.02)
    # Positive rate credits short; q resized to equity .99 on reversal.
    assert np.isclose(book[2,3],-.5*.03)
    assert book[-1,5]>book[1,5]

def test_path_labels_include_intrahour_adverse_excursion():
    price=np.full((865,2),100.)
    price[7]=[90.,110.]
    price[9]=[80.,100.]
    price[288]=[104.,102.]
    labels=path_labels(price,np.array([0]),np.array([4,24,72]),np.array([.5,.5]))
    assert np.isclose(labels[0,1],.03)
    assert np.isclose(labels[0,3],.10)
    assert np.isclose(labels[0,4],.03)

def test_risk_override_works_off_normal_review_clock():
    cfg={**CFG,'rebalance_hours':4,'uncertainty_penalty':0.}
    target,trade,reason=decision_path(np.array([.1,.1]),np.array([.01,.3]),np.array([.01,.01]),
        np.zeros(2),np.full(2,.001),np.array([0,1]),cfg)
    assert trade[0] and target[0]>.25
    assert trade[1] and reason[1]=='risk_override' and target[1]<=.25

def test_future_strong_signal_cannot_change_previous_decisions():
    args=[np.array([.02,-.02,.02]),np.full(3,.01),np.full(3,.01),np.full(3,.001),np.full(3,.001),np.array([0,4,8]),CFG]
    full=decision_path(*args)
    prefix=decision_path(*[a[:2] if isinstance(a,np.ndarray) else a for a in args])
    assert np.array_equal(full[0][:2],prefix[0])
    assert np.array_equal(full[1][:2],prefix[1])

def test_hourly_features_and_x2_are_prefix_invariant():
    rng=np.random.default_rng(5)
    idx=pd.date_range('2023-01-01',periods=1100,freq='h',tz='UTC')
    hourly={}
    for s in ['BTCUSDT','ETHUSDT']:
        hourly[s]=pd.DataFrame(dict(close=np.exp(np.cumsum(rng.normal(0,.002,len(idx)))),
            quote_volume=np.exp(rng.normal(10,.1,len(idx))),taker_buy_quote_volume=np.exp(rng.normal(9.3,.1,len(idx)))),index=idx)
    f,rv=market_features(hourly,['BTCUSDT','ETHUSDT'],[.5,.5],['BTCUSDT','ETHUSDT'])
    sub={s:v.iloc[:1000] for s,v in hourly.items()}
    prefix,_=market_features(sub,['BTCUSDT','ETHUSDT'],[.5,.5],['BTCUSDT','ETHUSDT'])
    assert np.allclose(f.iloc[:1000],prefix,equal_nan=True)
    assert path_features(f).shape[1]==100
    assert np.isclose(path_features(f).iloc[1000]['trend24_lag24'],f.iloc[976].trend24)

def test_soft_membership_is_geometry_not_future_label():
    x=np.array([[0.,0.],[2.,2.]])
    q,d=soft_membership(x,x.copy(),1.)
    assert np.allclose(q.sum(axis=1),1.)
    assert q[0,0]>q[0,1] and q[1,1]>q[1,0]
