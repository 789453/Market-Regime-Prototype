"""Independent raw-price labels, quantity accounting, maturity and freeze audit."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from crypto_market_beta_a import infer_bundle

def main():
    cfg=json.loads((ROOT/'configs/crypto_market_beta_a.json').read_text())
    out=ROOT/cfg['output_dir'];data=ROOT/cfg['data_dir']
    manifest=json.loads((out/'run_manifest.json').read_text())
    raw_ok=all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in manifest['raw_sources'])
    assert raw_ok
    audit=pd.read_csv(out/'fit_audit.csv')
    assert (pd.to_datetime(audit.max_label_maturity,utc=True)<pd.to_datetime(audit.forecast_start,utc=True)).all()
    pred=pd.read_parquet(data/'forward_predictions.parquet')
    frames={}
    for s in cfg['market_symbols']:
        f=pd.read_parquet(Path(cfg['raw_root'])/s/'5m.parquet')
        f.index=pd.to_datetime(f.open_time,unit='ms',utc=True)
        frames[s]=f
    main=pred[pred.model=='state_tree'].dropna(subset=['actual_return72']).sample(40,random_state=5)
    worst=0.
    for _,r in main.iterrows():
        start=r.available_at+pd.Timedelta('5min')
        p0=np.array([frames[s].loc[start,'open'] for s in cfg['market_symbols']])
        for h in cfg['horizons_hours']:
            p1=np.array([frames[s].loc[start+pd.Timedelta(hours=h),'open'] for s in cfg['market_symbols']])
            actual=np.array(cfg['market_weights'])@(p1/p0-1)
            worst=max(worst,abs(actual-r[f'actual_return{h}']))
        paths=np.column_stack([frames[s].loc[start:start+pd.Timedelta('24h'),'open'] for s in cfg['market_symbols']])
        returns=(paths/p0-1)@np.array(cfg['market_weights'])
        assert np.isclose(-min(0,returns.min()),r.actual_long_mae24)
        assert np.isclose(max(0,returns.max()),r.actual_short_mae24)
    assert worst<1e-12
    policies=pd.read_csv(out/'policy_scores.csv').policy.unique()
    fund=pd.read_parquet(data/'official_funding.parquet')
    max_cash_error=0.;max_quantity_change=0.;max_fee_error=0.;max_fund_error=0.;ledger_rows=0
    for policy in policies:
        g=pd.read_parquet(data/'positions_5m.parquet',filters=[('policy','==',policy)])
        g=g.sort_values('execution_at')
        ledger_rows+=len(g)
        e=np.r_[1.,g.equity.to_numpy()[:-1]]
        price=np.column_stack([frames[s].loc[pd.DatetimeIndex(g.execution_at),'open'] for s in cfg['market_symbols']])
        weights=g[[f'weight_{s}' for s in cfg['market_symbols']]].to_numpy()
        quantities=weights*e[:,None]/price
        old_quantities=np.vstack([np.zeros((1,len(cfg['market_symbols']))),quantities[:-1]])
        implied_price=np.sum(quantities*(np.column_stack([frames[s].loc[pd.DatetimeIndex(g.execution_at)+pd.Timedelta('5min'),'open'] for s in cfg['market_symbols']])-price),axis=1)/e
        max_cash_error=max(max_cash_error,float(np.max(np.abs(implied_price-g.price_return))))
        # No fee => no physical trade, except final liquidation at next mark.
        no_order=(g.turnover.to_numpy()[1:-1]<1e-10)
        changes=np.abs(np.diff(quantities,axis=0)[:-1]).max(axis=1)
        max_quantity_change=max(max_quantity_change,float(changes[no_order].max(initial=0)))
        fees=np.sum(np.abs(quantities-old_quantities)*price,axis=1)*6/1e4/e
        endpoint=np.array([frames[s].loc[g.execution_at.iloc[-1]+pd.Timedelta('5min'),'open'] for s in cfg['market_symbols']])
        fees[-1]+=np.sum(np.abs(quantities[-1])*endpoint)*6/1e4/e[-1]
        max_fee_error=max(max_fee_error,float(np.max(np.abs(fees-g.fee_return))))
        times=pd.DatetimeIndex(g.execution_at).append(pd.DatetimeIndex([g.execution_at.iloc[-1]+pd.Timedelta('5min')]))
        fund_rates=np.zeros((len(times),len(cfg['market_symbols'])))
        for j,s in enumerate(cfg['market_symbols']):
            f=fund[fund.symbol==s]
            loc=times.searchsorted(pd.DatetimeIndex(f.settlement_at))
            good=(loc<len(times))&(pd.DatetimeIndex(f.settlement_at)>=times[0])
            np.add.at(fund_rates[:,j],loc[good],f.last_funding_rate.to_numpy()[good])
        expected_fund=np.sum(old_quantities*price*fund_rates[:-1],axis=1)/e
        expected_fund[-1]+=np.sum(quantities[-1]*endpoint*fund_rates[-1])/e[-1]
        max_fund_error=max(max_fund_error,float(np.max(np.abs(expected_fund-g.funding_return))))
        assert np.allclose(g.net_return,g.price_return-g.fee_return-g.funding_return,atol=2e-14)
        assert np.isfinite(g.gross).all()  # target cap and natural drift differ.
    assert max_cash_error<1e-12 and max_quantity_change<1e-12 and max_fee_error<1e-12 and max_fund_error<1e-12
    frozen=joblib.load(out/'frozen_model.joblib')
    row=pd.read_parquet(data/'latest_features.parquet').set_index('available_at')
    output,_,_=infer_bundle(frozen,row)
    source=json.loads((out/'latest_historical_signal.json').read_text())
    # Known hourly scale belongs to same last historical snapshot.
    f=pd.read_parquet(data/'market_features.parquet')
    all_close=pd.concat({s:np.log(pd.read_parquet(Path(cfg['raw_root'])/s/'1h.parquet').close) for s in cfg['market_symbols']},axis=1)
    rv=np.sqrt((all_close.diff()@np.array(cfg['market_weights'])).pow(2).rolling(168).mean()).iloc[-2]
    # Last x snapshot may be one hour before data end; locate by hourly close.
    hourly=pd.read_parquet(Path(cfg['raw_root'])/cfg['market_symbols'][0]/'1h.parquet')
    at=pd.DatetimeIndex(pd.to_datetime(hourly.open_time,unit='ms',utc=True))+pd.Timedelta('1h')
    scale_series=pd.Series(np.sqrt((all_close.diff()@np.array(cfg['market_weights'])).pow(2).rolling(168).mean()).to_numpy(),index=at)
    rv=float(scale_series.loc[row.index[0]])
    factors=np.r_[rv*np.sqrt([4,24,72,24,24,24]),1.,1.,rv*np.sqrt(24)]
    error=max(abs(output[m][0,j]*factors[j]-source['predictions'][m][task]) for m in output for j,task in enumerate(['return4','return24','return72','long_mae24','short_mae24','rv24','prob_down24','prob_tail24','return24_q10']))
    assert error<1e-12
    result=dict(raw_hashes_unchanged=raw_ok,label_samples=len(main),max_price_label_error=worst,
        ledger_rows=ledger_rows,max_independent_price_pnl_error=max_cash_error,
        max_no_order_quantity_change=max_quantity_change,max_fee_identity_error=max_fee_error,
        max_independent_funding_cashflow_error=max_fund_error,
        frozen_inference_max_error=error,all_label_maturity_pass=True)
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
