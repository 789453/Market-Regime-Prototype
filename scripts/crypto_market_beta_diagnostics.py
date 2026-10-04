"""Monthly drift, conditional errors, episode censoring and capacity proxies."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/crypto_market_beta_a.json').read_text())
OUT=ROOT/CFG['output_dir'];DATA=ROOT/CFG['data_dir']

def save(frame,name):
    frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')

def main():
    daily=pd.read_csv(OUT/'portfolio_daily.csv',parse_dates=['day'])
    daily['month']=daily.day.dt.strftime('%Y-%m')
    monthly=daily.groupby(['policy','fee_bp_side','month'],as_index=False).agg(
        net_return=('net_return',lambda x:(1+x).prod()-1),mean_beta=('net_beta','mean'),
        mean_gross=('gross','mean'),fee_bp=('fee_bp','sum'),funding_bp=('funding_bp','sum'),
        price_bp=('price_bp','sum'),long_bp=('long_bp','sum'),short_bp=('short_bp','sum'))
    save(monthly,'monthly_policy_cards.csv')
    pred=pd.read_parquet(DATA/'forward_predictions.parquet')
    pred['month']=pred.available_at.dt.strftime('%Y-%m')
    conditional=[]
    for (model,fold,state),g in pred.groupby(['model','fold','state']):
        valid=g.actual_return24_z.notna()
        y=g.loc[valid,'actual_return24_z'];p=g.loc[valid,'pred_return24_z']
        conditional.append(dict(model=model,fold=fold,state=state,hours=int(valid.sum()),
            independent_calendar_days=g.loc[valid,'available_at'].dt.floor('D').nunique(),
            mse_z=float(((y-p)**2).mean()),ic=float(y.corr(p)) if p.std()>1e-12 and y.std()>1e-12 else np.nan,
            direction_bias_bp=float((g.loc[valid,'pred_return24']-g.loc[valid,'actual_return24']).mean()*1e4),
            down_brier=float(((g.loc[valid,'pred_prob_down24']-g.loc[valid,'actual_prob_down24'])**2).mean()),
            tail_brier=float(((g.loc[valid,'pred_prob_tail24']-g.loc[valid,'actual_prob_tail24'])**2).mean())))
    save(pd.DataFrame(conditional),'conditional_prediction_cards.csv')
    # Production training support at each already-frozen outer fit. Descriptive
    # drift shares are not fed back to change historical policies.
    import joblib
    from src.crypto.market_beta import path_features
    f=path_features(pd.read_parquet(DATA/'market_features.parquet'),CFG['path_lags_hours']).dropna()
    drift=[]
    for fold,g in pred[pred.model=='state_tree'].groupby('fold'):
        bundle=joblib.load(OUT/f'model_{fold}.joblib')
        x=f.reindex(pd.DatetimeIndex(g.available_at))
        z=bundle['scaler'].transform(x)
        for month in g.month.unique():
            selected=g.month.to_numpy()==month
            drift.append(dict(fold=fold,month=month,hours=int(selected.sum()),
                mean_abs_training_z=float(np.abs(z[selected]).mean()),
                coordinate_share_over_3sd=float((np.abs(z[selected])>3).mean()),
                snapshot_share_any_over_5sd=float((np.abs(z[selected])>5).any(axis=1).mean()),
                purpose='input support diagnostic; no hindsight policy gate'))
    save(pd.DataFrame(drift),'input_drift_cards.csv')
    calibration=[]
    for (model,fold),g in pred.groupby(['model','fold']):
        for task in ['prob_down24','prob_tail24']:
            valid=g[f'actual_{task}'].notna()
            p=g.loc[valid,f'pred_{task}'].to_numpy();y=g.loc[valid,f'actual_{task}'].to_numpy()
            bins=np.clip((p*10).astype(int),0,9)
            for k in range(10):
                selected=bins==k
                if selected.any():
                    calibration.append(dict(model=model,fold=fold,task=task,probability_bin=k,
                        hours=int(selected.sum()),predicted=float(p[selected].mean()),actual=float(y[selected].mean())))
    save(pd.DataFrame(calibration),'probability_calibration.csv')
    decision=pd.read_parquet(DATA/'decision_rows.parquet')
    episodes=[]
    terminal=pd.Timestamp('2026-09-24T23:55:00Z')
    for policy,g in decision.groupby('policy'):
        g=g.sort_values('available_at').reset_index(drop=True)
        signs=np.sign(g.target_beta.to_numpy())
        edges=np.flatnonzero(np.r_[True,signs[1:]!=signs[:-1]])
        for i,left in enumerate(edges):
            if signs[left]==0:
                continue
            right=edges[i+1] if i+1<len(edges) else len(g)
            entry=g.available_at.iloc[left]+pd.Timedelta('5min')
            exit=g.available_at.iloc[right]+pd.Timedelta('5min') if right<len(g) else terminal
            episodes.append(dict(policy=policy,entry_at=entry,exit_at=exit,sign=int(signs[left]),
                duration_hours=(exit-entry).total_seconds()/3600,orders_inside=int(g.order.iloc[left:right].sum()),
                terminal_censored=right==len(g),mean_target=float(g.target_beta.iloc[left:right].mean())))
    pd.DataFrame(episodes).to_parquet(DATA/'episodes.parquet',index=False)
    summary=pd.DataFrame(episodes).groupby('policy',as_index=False).agg(
        episodes=('duration_hours','size'),median_hours=('duration_hours','median'),mean_hours=('duration_hours','mean'),
        max_hours=('duration_hours','max'),fraction_over120h=('duration_hours',lambda s:(s>120).mean()),censored=('terminal_censored','sum'))
    save(summary,'episode_summary.csv')
    raw={}
    for s in CFG['market_symbols']:
        r=pd.read_parquet(Path(CFG['raw_root'])/s/'5m.parquet',columns=['open_time','open','quote_volume'])
        r.index=pd.to_datetime(r.open_time,unit='ms',utc=True)
        raw[s]=r
    capacities=[]
    for policy in ['trend168','trend_multi','state_ridge','state_tree','x2_tree','x2_soft_local']:
        g=pd.read_parquet(DATA/'positions_5m.parquet',filters=[('policy','==',policy)]).sort_values('execution_at')
        idx=pd.DatetimeIndex(g.execution_at)
        start_equity=np.r_[1.,g.equity.to_numpy()[:-1]]
        prices=np.column_stack([raw[s].open.reindex(idx) for s in CFG['market_symbols']])
        q=g[[f'weight_{s}' for s in CFG['market_symbols']]].to_numpy()*start_equity[:,None]/prices
        old=np.vstack([np.zeros((1,2)),q[:-1]])
        traded=np.abs(q-old)*prices
        # Last row also charges terminal closure, recorded as a separate
        # event with its own prior complete 5m volume, not the opening volume.
        final_prices=np.array([raw[s].loc[terminal,'open'] for s in CFG['market_symbols']])
        traded=np.vstack([traded,np.abs(q[-1])*final_prices])
        trade_times=idx.append(pd.DatetimeIndex([terminal]))
        volume=np.column_stack([raw[s].quote_volume.reindex(trade_times-pd.Timedelta('5min')) for s in CFG['market_symbols']])
        selected=traded>1e-10
        for capital in [1e6,1e7,1e8]:
            observable=selected&(volume>0)&np.isfinite(volume)
            participation=(traded[observable]*capital/volume[observable])
            capacities.append(dict(policy=policy,initial_capital_usdt=capital,orders_legs=int(selected.sum()),
                undefined_volume_legs=int((selected&~observable).sum()),
                median_previous5m_volume_fraction=float(np.median(participation)),p95_fraction=float(np.quantile(participation,.95)),
                max_fraction=float(participation.max()),fraction_over1pct=float((participation>.01).mean()),
                interpretation='prior completed quote volume proxy; no depth, fill or impact certification'))
    save(pd.DataFrame(capacities),'capacity_proxy.csv')
    # Numerical-fix iteration should remain identical after adding passive
    # probability/quantile tasks and diagnostics.
    before=pd.read_csv(OUT/'iteration_02_policy_scores.csv')
    after=pd.read_csv(OUT/'policy_scores.csv')
    joined=before.merge(after,on=['policy','fee_bp_side','phase'],suffixes=('_old','_new'))
    max_change=float(np.max(np.abs(joined.return_total_old-joined.return_total_new)))
    assert max_change<1e-10
    (OUT/'iteration_consistency.json').write_text(json.dumps(dict(common_policy_rows=len(joined),
        max_core_policy_return_change_after_passive_task_extension=max_change)),encoding='utf-8')
    print('Monthly, conditional, episode, capacity and iteration diagnostics complete.')

if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    main()
