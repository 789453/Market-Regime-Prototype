"""Read-only completed-bar inference; explicit paper actions, never orders."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from crypto_market_beta_a import TASKS,infer_bundle
from src.crypto.market_beta import decision_path,market_features,path_features

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--as-of',help='UTC availability timestamp; incomplete bars are rejected')
    parser.add_argument('--previous-beta',type=float,default=0.,help='paper planner prior target; actual exchange position not queried')
    parser.add_argument('--raw-root',help='read-only newer source root, same schema and symbols')
    parser.add_argument('--funding-file',help='newer verified settlement parquet; no future rates in features')
    parser.add_argument('--output',default='reports/crypto/market_beta_a_v1/shadow_historical_preview.json')
    args=parser.parse_args()
    cfg=json.loads((ROOT/'configs/crypto_market_beta_a.json').read_text())
    hourly={}
    for s in cfg['context_symbols']:
        f=pd.read_parquet(Path(args.raw_root or cfg['raw_root'])/s/'1h.parquet')
        f.index=pd.to_datetime(f.open_time,unit='ms',utc=True)+pd.Timedelta('1h')
        times=pd.DatetimeIndex(f.index).as_unit('ns')
        if not times.is_monotonic_increasing or times.duplicated().any() or not np.all(np.diff(times.asi8)==pd.Timedelta('1h').value):
            raise ValueError('source clock gap/duplicate: '+s)
        hourly[s]=f
    latest=min(f.index.max() for f in hourly.values())
    stamp=pd.Timestamp(args.as_of) if args.as_of else latest
    if stamp.tz is None:
        raise ValueError('--as-of must include timezone')
    stamp=stamp.tz_convert('UTC')
    if stamp>latest or stamp.floor('h')!=stamp or stamp>pd.Timestamp(datetime.now(timezone.utc)):
        raise ValueError('as-of unavailable or incomplete hourly boundary')
    hourly={s:f[f.index<=stamp] for s,f in hourly.items()}
    funding_file=Path(args.funding_file) if args.funding_file else ROOT/cfg['data_dir']/'official_funding.parquet'
    known=None
    if funding_file.exists():
        funds=pd.read_parquet(funding_file)
        sums=[]
        for s,w in zip(cfg['market_symbols'],cfg['market_weights']):
            f=funds[(funds.symbol==s)&(funds.settlement_at+pd.Timedelta('1h')<=stamp)].sort_values('settlement_at')
            lag=pd.Series(f.last_funding_rate.to_numpy(),index=pd.DatetimeIndex(f.settlement_at)+pd.Timedelta('1h'))
            idx=hourly[s].index
            sums.append(lag.reindex(idx.union(lag.index)).ffill().reindex(idx).fillna(0)*w)
        known=sum(sums)
    features,rv=market_features(hourly,cfg['market_symbols'],cfg['market_weights'],cfg['context_symbols'],known)
    x=path_features(features,cfg['path_lags_hours']).tail(1)
    if x.isna().any().any():
        raise ValueError('insufficient warmup / missing data')
    path=ROOT/cfg['output_dir']/'frozen_model.joblib'
    bundle=joblib.load(path)
    if stamp<pd.Timestamp(bundle['fit_cutoff']):
        mode='historical reconstruction; frozen model contains later historical information'
    elif stamp<pd.Timestamp('2026-10-05',tz='UTC'):
        mode='historical preview; not new-date confirmation'
    elif stamp+pd.Timedelta('4h5min')<=pd.Timestamp(datetime.now(timezone.utc)):
        mode='backfilled historical preview; shortest future label already matured before recording'
    else:
        mode='paper sequential candidate; preserve this output before observing future labels'
    p,_,_=infer_bundle(bundle,x)
    scale=np.r_[float(rv.iloc[-1])*np.sqrt([4,24,72,24,24,24]),1.,1.,float(rv.iloc[-1])*np.sqrt(24)]
    output={}
    for m,values in p.items():
        v=values[0]*scale
        mu=np.dot(v[:3]*np.array([6.,1.,1/3]),[.2,.5,.3])-features.known_funding.iloc[-1]/1e4*3
        uncertainty=float(rv.iloc[-1])*np.sqrt(24)*(1+np.std(values[0,:3]))
        target,trade,reason=decision_path(np.array([mu]),np.array([v[3]]),np.array([v[4]]),np.array([uncertainty]),np.array([rv.iloc[-1]]),np.array([stamp.hour]),cfg,initial_beta=args.previous_beta)
        output[m]=dict(predictions={k:float(v[j]) for j,k in enumerate(TASKS)},prior_planned_beta=args.previous_beta,
            proposed_beta=float(target[0]),paper_order=bool(trade[0]),reason=str(reason[0]),mu24_equivalent=float(mu))
    result=dict(available_at=str(stamp),recorded_at=datetime.now(timezone.utc).isoformat(),earliest_execution_at=str(stamp+pd.Timedelta('5min')),status=mode,
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),live_order=False,candidates=output)
    destination=ROOT/args.output
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and mode.startswith('paper sequential'):
        raise ValueError('sequential prediction archive is immutable; choose a new dated output')
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(available_at=result['available_at'],status=mode,output=str(destination)),ensure_ascii=False))

if __name__=='__main__':
    main()
