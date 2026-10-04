"""Progressive market Beta research; historical development, never untouched OOS."""
from __future__ import annotations
import hashlib
import argparse
import json
import platform
import sys
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.crypto.market_beta import (FEATURES,BOOK_COLUMNS,block_interval,decision_path,
    feature_registry,market_features,path_features,path_labels,quantity_ledger,soft_membership)

CFG=json.loads((ROOT/'configs/crypto_market_beta_a.json').read_text())
OUT=ROOT/CFG['output_dir']
DATA=ROOT/CFG['data_dir']
TASKS=['return4','return24','return72','long_mae24','short_mae24','rv24','prob_down24','prob_tail24','return24_q10']
MODELS=['historical_mean','state_ridge','state_tree','x2_tree','x2_no_internal','x2_soft_local']

def save_csv(frame,name):
    frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')

def load_inputs():
    hourly, five, raw_manifest={}, {}, []
    cutoff=pd.Timestamp(CFG['historical_end_exclusive'])
    for symbol in CFG['context_symbols']:
        for freq in ['1h','5m']:
            if freq=='5m' and symbol not in CFG['market_symbols']:
                continue
            path=Path(CFG['raw_root'])/symbol/f'{freq}.parquet'
            frame=pd.read_parquet(path)
            stamp=pd.DatetimeIndex(pd.to_datetime(frame.open_time,unit='ms',utc=True)).as_unit('ns')
            if not stamp.is_monotonic_increasing or stamp.duplicated().any():
                raise ValueError('invalid source clock')
            step=pd.Timedelta(freq).value
            if not np.all(np.diff(stamp.asi8)==step):
                raise ValueError('source gap')
            frame.index=stamp+pd.Timedelta(freq) if freq=='1h' else stamp
            frame=frame[frame.index<=cutoff] if freq=='1h' else frame[frame.index<cutoff]
            if freq=='1h':
                hourly[symbol]=frame
            else:
                five[symbol]=frame
            raw_manifest.append(dict(path=str(path),bytes=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                first=str(frame.index.min()),last=str(frame.index.max()),rows=len(frame)))
    price=pd.concat({s:five[s].open for s in CFG['market_symbols']},axis=1)
    if price.isna().any().any():
        raise ValueError('main-market missing price')
    funding_file=DATA/'official_funding.parquet'
    funds=pd.read_parquet(funding_file) if funding_file.exists() else pd.DataFrame()
    funded=np.zeros(price.shape)
    known=pd.Series(0.,index=price.index)
    fund_audit=[]
    if len(funds):
        if funds.duplicated(['symbol','settlement_at']).any():
            raise ValueError('duplicate funding event')
        basket=[]
        for j,s in enumerate(CFG['market_symbols']):
            f=funds[funds.symbol==s].sort_values('settlement_at')
            settlement=pd.DatetimeIndex(f.settlement_at).as_unit('ns')
            # Actual event time may carry a few milliseconds. Assign to the
            # first grid point at/after settlement, never retroactively.
            pos=price.index.searchsorted(settlement)
            valid=pos<len(price)
            np.add.at(funded[:,j],pos[valid],f.last_funding_rate.to_numpy()[valid])
            lag=pd.Series(f.last_funding_rate.to_numpy(),index=settlement+pd.Timedelta('1h'))
            union=price.index.union(lag.index)
            basket.append(lag.reindex(union).ffill().reindex(price.index).fillna(0)*CFG['market_weights'][j])
            dt=np.diff(settlement.asi8)/3.6e12
            expected=np.asarray(f.funding_interval_hours.iloc[1:],dtype=float)
            mismatch=np.abs(dt-expected)>1/60
            fund_audit.append(dict(symbol=s,events=len(f),first=str(settlement.min()),last=str(settlement.max()),
                gaps_over_reported_interval=int(mismatch.sum()),max_interval_hours=float(dt.max()),
                interval_hours=';'.join(map(str,sorted(f.funding_interval_hours.unique()))),
                settlement_mark='5m trade-open proxy; exact mark unavailable',
                complete_through_end=bool(settlement.max()>=price.index.max()-pd.Timedelta('8h'))))
        known=sum(basket)
    save_csv(pd.DataFrame(fund_audit),'funding_coverage.csv')
    return hourly,five,price,funded,known,raw_manifest

def make_tree(quantile=None):
    options={} if quantile is None else dict(objective='quantile',alpha=quantile)
    return lgb.LGBMRegressor(**options,n_estimators=CFG['tree_estimators'],num_leaves=CFG['tree_leaves'],
        learning_rate=.035,min_child_samples=CFG['tree_min_child_samples'],reg_lambda=20.,
        colsample_bytree=1.,subsample=1.,random_state=CFG['seed'],n_jobs=4,verbosity=-1,
        deterministic=True,force_col_wise=True)

def fit_heads(x,y,kind):
    lo,hi=np.quantile(y,[.005,.995],axis=0)
    clipped=np.clip(y,lo,hi)
    clipped[:,6:8]=y[:,6:8]  # Brier heads retain actual binary events.
    heads=[]
    for j in range(y.shape[1]):
        model=make_tree(.1) if j==8 else Ridge(alpha=CFG['ridge_alpha']) if kind=='ridge' else make_tree()
        model.fit(x,clipped[:,j])
        heads.append(model)
    return heads

def predict_heads(heads,x):
    return np.column_stack([m.booster_.predict(x,num_threads=4) if hasattr(m,'booster_') else m.predict(x) for m in heads])

def fit_bundle(x,y,times):
    scaler=StandardScaler().fit(x)
    z=np.clip(scaler.transform(x),-8,8)
    # Inner chronological forecast residuals, not fitted residuals, calibrate
    # a six-region local correction. All calibration labels mature externally.
    split=times[-1]-pd.Timedelta('90D')
    inner_train=times+pd.Timedelta('72h5min')<split
    calibration=times>=split
    geometry_scaler=StandardScaler().fit(x.loc[inner_train])
    geometry_z=np.clip(geometry_scaler.transform(x),-8,8)
    kmeans=KMeans(n_clusters=CFG['prototype_count'],random_state=CFG['seed'],n_init=5)
    kmeans.fit(geometry_z[inner_train][::6])
    centers=kmeans.cluster_centers_
    d=((geometry_z[inner_train][::6,None,:]-centers[None,:,:])**2).sum(axis=2)
    sorted_d=np.sort(d,axis=1)
    temperature=max(np.median(sorted_d[:,1]-sorted_d[:,0]),1.)
    inner_heads=fit_heads(geometry_z[inner_train],y[inner_train],'tree')
    residual=y[calibration]-predict_heads(inner_heads,geometry_z[calibration])
    q,_=soft_membership(geometry_z[calibration],centers,temperature)
    correction=q.T@residual/(q.sum(axis=0)[:,None]+CFG['prototype_shrinkage_hours'])
    correction[:,8]=0.  # Mean-residual correction is invalid for a quantile.
    columns=np.array([not any(k in name for k in ('breadth24','dispersion24')) for name in x.columns])
    bundle=dict(scaler=scaler,columns=list(x.columns),mean=y.mean(axis=0),
        state_ridge=fit_heads(z[:,:20],y,'ridge'),state_tree=fit_heads(z[:,:20],y,'tree'),
        x2_tree=fit_heads(z,y,'tree'),x2_no_internal=fit_heads(z[:,columns],y,'tree'),
        no_internal_columns=columns,geometry_scaler=geometry_scaler,centers=centers,temperature=temperature,correction=correction,
        fit_first=str(times[0]),fit_last_feature=str(times[-1]),
        fit_last_label_maturity=str(times[-1]+pd.Timedelta('72h5min')),
        nn=NearestNeighbors(n_neighbors=10).fit(geometry_z[inner_train][::6]))
    bundle['mean'][8]=np.quantile(y[:,8],.1)
    return bundle

def infer_bundle(bundle,x):
    if list(x.columns)!=bundle['columns']:
        raise ValueError('feature order changed')
    z=np.clip(bundle['scaler'].transform(x),-8,8)
    p={
        'historical_mean':np.tile(bundle['mean'],(len(x),1)),
        'state_ridge':predict_heads(bundle['state_ridge'],z[:,:20]),
        'state_tree':predict_heads(bundle['state_tree'],z[:,:20]),
        'x2_tree':predict_heads(bundle['x2_tree'],z),
        'x2_no_internal':predict_heads(bundle['x2_no_internal'],z[:,bundle['no_internal_columns']])}
    geometry_z=np.clip(bundle['geometry_scaler'].transform(x),-8,8)
    q,d=soft_membership(geometry_z,bundle['centers'],bundle['temperature'])
    p['x2_soft_local']=p['x2_tree']+q@bundle['correction']
    for model in p:
        p[model][:,3:6]=np.maximum(.05,p[model][:,3:6])
        p[model][:,6:8]=np.clip(p[model][:,6:8],.001,.999)
    return p,q,d

def forecasts(features,rv,prices):
    x=path_features(features,CFG['path_lags_hours']).dropna()
    x=x[x.index+pd.Timedelta('5min')<=prices.index[-1]]
    e=prices.index.get_indexer(x.index+pd.Timedelta('5min'))
    labels=path_labels(prices.to_numpy(),e,np.asarray(CFG['horizons_hours']),np.asarray(CFG['market_weights']))
    scales=rv.reindex(x.index).to_numpy()[:,None]*np.sqrt(np.array([4,24,72,24,24,24]))
    valid24=np.isfinite(labels[:,1])
    probabilities=np.column_stack([(labels[:,1]<0).astype(float),(labels[:,1]<-1.5*scales[:,1]).astype(float)])
    probabilities[~valid24]=np.nan
    labels=np.column_stack([labels,probabilities,labels[:,1]])
    scales=np.column_stack([scales,np.ones((len(scales),2)),scales[:,1]])
    y=labels/scales
    rows=[]
    audit=[]
    identity=[]
    medoids=[]
    support=[]
    previous=None
    starts=[pd.Timestamp(s,tz='UTC') for s in CFG['fold_starts']]
    for fold,start in enumerate(starts):
        stop=starts[fold+1] if fold+1<len(starts) else pd.Timestamp(CFG['historical_end_exclusive'])
        train=(x.index+pd.Timedelta('72h5min')<start)&np.isfinite(y).all(axis=1)
        forward=(x.index>=start)&(x.index<stop)
        print(f'Fit {start.date()}: {train.sum()} training hours; {forward.sum()} forward hours',flush=True)
        bundle=fit_bundle(x.loc[train],y[train],x.index[train])
        bundle['forecast_start']=str(start)
        p,q,d=infer_bundle(bundle,x.loc[forward])
        a=dict(fold=str(start.date()),train_rows=int(train.sum()),forward_rows=int(forward.sum()),
            fit_last=bundle['fit_last_feature'],max_label_maturity=bundle['fit_last_label_maturity'],forecast_start=str(start),
            purge_hours=72+5/60,parameters_selected_from_current_fold=False)
        audit.append(a)
        if not pd.Timestamp(a['max_label_maturity'])<start:
            raise AssertionError('unmatured label in fit')
        joblib.dump(bundle,OUT/f'model_{start.date()}.joblib',compress=3)
        if previous is not None:
            common=x.loc[train].tail(1440)
            _,_,prev_d=infer_bundle(previous,common)
            _,_,cur_d=infer_bundle(bundle,common)
            identity.append(dict(fold=str(start.date()),common_hours=len(common),ari=adjusted_rand_score(prev_d.argmin(axis=1),cur_d.argmin(axis=1))))
        z=np.clip(bundle['geometry_scaler'].transform(x.loc[train]),-8,8)
        query=x.loc[forward].iloc[::24]
        query_z=np.clip(bundle['geometry_scaler'].transform(query),-8,8)
        distances,neighbors=bundle['nn'].kneighbors(query_z)
        inner_end=x.index[train][-1]-pd.Timedelta('90D')
        anchor_times=x.index[train][x.index[train]+pd.Timedelta('72h5min')<inner_end][::6]
        for j,t in enumerate(query.index):
            support.append(dict(fold=str(start.date()),available_at=str(t),nearest_distance=float(distances[j,0]),
                tenth_distance=float(distances[j,-1]),nearest_real_history=str(anchor_times[neighbors[j,0]]),
                neighbor_unique_days=len(set(anchor_times[neighbors[j]].floor('D'))),
                interpretation='geometry support only; not prediction confidence'))
        medoid_limit=x.index[train][-1]-pd.Timedelta('90D')
        valid_medoid=x.index[train]+pd.Timedelta('72h5min')<medoid_limit
        for k,c in enumerate(bundle['centers']):
            distances=((z-c)**2).sum(axis=1)
            distances[~valid_medoid]=np.inf
            j=int(distances.argmin())
            medoids.append(dict(fold=str(start.date()),region=k,available_at=str(x.index[train][j]),
                squared_distance=float(distances[j]),local_return24_correction_z=float(bundle['correction'][k,1]),
                prototype_role='unverified soft geometry / local residual correction',
                **{f'path_trend24_lag{lag}':float(x.loc[train].iloc[j][f'trend24_lag{lag}']) for lag in CFG['path_lags_hours']}))
        for model,pred in p.items():
            frame=pd.DataFrame(index=x.index[forward])
            frame['model']=model
            frame['fold']=str(start.date())
            frame['history_status']=np.where(frame.index.year>=2026,'reused_exploration','development_forward')
            frame['rv_hour']=rv.reindex(frame.index)
            frame['region']=d.argmin(axis=1)
            frame['region_distance']=d.min(axis=1)
            for j,task in enumerate(TASKS):
                frame[f'pred_{task}_z']=pred[:,j]
                frame[f'pred_{task}']=pred[:,j]*scales[forward,j]
                frame[f'actual_{task}_z']=y[forward,j]
                frame[f'actual_{task}']=labels[forward,j]
            frame['trend_signal'] = np.tanh((features.reindex(frame.index).trend24+features.reindex(frame.index).trend168)/2)
            frame['slow_signal'] = np.tanh(features.reindex(frame.index).trend168)
            frame['short_signal'] = np.tanh(features.reindex(frame.index).trend24)
            for name in ['trend24','trend168','efficiency24','risk_ratio','breadth24','known_funding']:
                frame[name]=features.reindex(frame.index)[name]
            risk_level=np.quantile(features.reindex(x.index).loc[train,'risk_ratio'],.8)
            frame['state']=np.where(frame.risk_ratio>risk_level,'risk_expansion',np.where(frame.trend24>0.5,'uptrend',np.where(frame.trend24<-.5,'downtrend','mixed')))
            frame['rebound_flag']=(features.reindex(frame.index).drawdown168<-.5)&(features.reindex(frame.index).acceleration24>.5)
            rows.append(frame.reset_index(names='available_at'))
        previous=bundle
    save_csv(pd.DataFrame(audit),'fit_audit.csv')
    save_csv(pd.DataFrame(identity),'prototype_identity.csv')
    save_csv(pd.DataFrame(medoids),'prototype_medoids.csv')
    save_csv(pd.DataFrame(support),'prototype_support.csv')
    result=pd.concat(rows,ignore_index=True)
    result.to_parquet(DATA/'forward_predictions.parquet',index=False)
    # Production-like freeze uses only labels matured by the recorded data end.
    cutoff=pd.Timestamp(CFG['historical_end_exclusive'])
    train=(x.index+pd.Timedelta('72h5min')<cutoff)&np.isfinite(y).all(axis=1)
    frozen=fit_bundle(x.loc[train],y[train],x.index[train])
    frozen['fit_cutoff']=str(cutoff)
    frozen['frozen_on']='2026-10-04'
    joblib.dump(frozen,OUT/'frozen_model.joblib',compress=3)
    latest=x.tail(1)
    p,_,_=infer_bundle(frozen,latest)
    snap={m:{task:float(p[m][0,j]*scales[-1,j]) for j,task in enumerate(TASKS)} for m in p}
    (OUT/'latest_historical_signal.json').write_text(json.dumps(dict(available_at=str(latest.index[0]),execution_at=str(latest.index[0]+pd.Timedelta('5min')),status='historical snapshot; not current advice',predictions=snap),ensure_ascii=False,indent=2),encoding='utf-8')
    latest.reset_index(names='available_at').to_parquet(DATA/'latest_features.parquet',index=False)
    return result,x

def prediction_scores(pred):
    rows=[]
    daily=[]
    calibration=[]
    for (model,fold),g in pred.groupby(['model','fold']):
        for task in TASKS:
            p=g[f'pred_{task}_z'].to_numpy()
            y=g[f'actual_{task}_z'].to_numpy()
            good=np.isfinite(y)
            p,y=p[good],y[good]
            mse=np.mean((p-y)**2)
            loss=(.1-(y<p).astype(float))*(y-p) if task=='return24_q10' else (p-y)**2
            ic=np.corrcoef(p,y)[0,1] if np.std(p)>1e-10 else np.nan
            rows.append(dict(model=model,fold=fold,task=task,n=len(y),mse_z=mse,ic=ic,
                bias_z=float((p-y).mean()),proper_score=float(loss.mean()),
                score_type='pinball_q10' if task=='return24_q10' else 'brier' if task.startswith('prob') else 'mse',
                q10_breach_rate=float((y<p).mean()) if task=='return24_q10' else np.nan,
                mse_bp2=float(np.mean((g.loc[good,f'pred_{task}']-g.loc[good,f'actual_{task}'])**2)*1e8)))
            d=g.loc[good,['available_at','state']].copy()
            d['model'],d['task'],d['loss']=model,task,loss
            d['day']=d.available_at.dt.floor('D')
            daily.append(d.groupby(['day','model','task','state'],as_index=False).agg(loss=('loss','mean'),hours=('loss','size')))
            if task=='return24':
                # Rank bins solely as descriptive forward calibration; never a
                # source of action thresholds or subsequent fitting.
                bins=pd.qcut(pd.Series(p).rank(method='first'),5,labels=False).to_numpy()
                for k in range(5):
                    mask=bins==k
                    calibration.append(dict(model=model,fold=fold,bin=k+1,n=int(mask.sum()),
                        predicted_bp=float(g.loc[good,f'pred_{task}'].to_numpy()[mask].mean()*1e4),
                        actual_bp=float(g.loc[good,f'actual_{task}'].to_numpy()[mask].mean()*1e4)))
    scores=pd.DataFrame(rows)
    save_csv(scores,'prediction_scores.csv')
    save_csv(pd.concat(daily),'daily_prediction_loss.csv')
    save_csv(pd.DataFrame(calibration),'direction_calibration.csv')
    comparisons=[]
    d=pd.concat(daily)
    d['weighted_loss']=d.loss*d.hours
    d=d.groupby(['day','model','task'],as_index=False).agg(weighted_loss=('weighted_loss','sum'),hours=('hours','sum'))
    d['loss']=d.weighted_loss/d.hours
    for task in TASKS:
        pivot=d[d.task==task].pivot(index='day',columns='model',values='loss')
        for candidate,reference in [('state_ridge','historical_mean'),('state_tree','historical_mean'),('state_tree','state_ridge'),('x2_tree','state_tree'),('x2_soft_local','x2_tree'),('x2_no_internal','x2_tree')]:
            for phase,mask in [('development',pivot.index.year<2026),('reused_2026',pivot.index.year>=2026),('all',np.ones(len(pivot),bool))]:
                mean,low,high=block_interval((pivot[reference]-pivot[candidate]).loc[mask],CFG['bootstrap_block_days'],CFG['bootstrap_repetitions'])
                comparisons.append(dict(task=task,candidate=candidate,reference=reference,phase=phase,loss_improvement=mean,low=low,high=high))
    save_csv(pd.DataFrame(comparisons),'prediction_increment.csv')
    return scores

def expand_orders(index,clock,target,trade,delay=5):
    pos=index.get_indexer(clock+pd.Timedelta(minutes=delay))
    if (pos<0).any():
        raise ValueError('missing execution timestamp')
    out=np.zeros(len(index)-1)
    orders=np.zeros(len(index)-1,dtype=np.bool_)
    good=pos<len(orders)
    orders[pos[good]]=trade[good]
    out[pos[good]]=target[good]
    return out,orders

def make_book(price,funding,clock,target,trade,fee,delay=5,weights=None):
    target5,order5=expand_orders(price.index,clock,target,trade,delay)
    values=quantity_ledger(price.to_numpy(),target5,order5,funding,
        np.asarray(CFG['market_weights'] if weights is None else weights),fee)
    return pd.DataFrame(values,index=price.index[:-1],columns=BOOK_COLUMNS+[f'weight_{s}' for s in CFG['market_symbols']])

def book_stats(book,policy,fee,phase):
    d=(1+book.net_return).resample('D').prod()-1
    equity=(1+book.net_return).cumprod()
    dd=equity/equity.cummax().clip(lower=1.)-1
    vol=d.std()*np.sqrt(365)
    sign=np.sign(book.net_beta.to_numpy())
    active=np.abs(book.net_beta.to_numpy())>.05
    edges=np.flatnonzero(np.r_[True,(sign[1:]!=sign[:-1])])
    lengths=np.diff(np.r_[edges,len(sign)])/12
    active_lengths=lengths[sign[edges]!=0]
    return dict(policy=policy,fee_bp_side=fee,phase=phase,return_total=float(equity.iloc[-1]-1),max_drawdown=float(dd.min()),
        daily_sharpe=float(d.mean()/d.std()*np.sqrt(365)) if d.std()>0 else np.nan,annual_vol=float(vol),
        mean_net_beta=float(book.net_beta.mean()),mean_gross=float(book.gross.mean()),
        participation=float(active.mean()),long_time=float((book.net_beta>.05).mean()),short_time=float((book.net_beta<-.05).mean()),
        turnover=float(book.turnover.sum()),trade_rows=int((book.turnover>1e-8).sum()),
        mean_sign_episode_hours=float(active_lengths.mean()) if len(active_lengths) else 0.,
        median_sign_episode_hours=float(np.median(active_lengths)) if len(active_lengths) else 0.,
        fee_bp_sum=float(book.fee_return.sum()*1e4),funding_bp_sum=float(book.funding_return.sum()*1e4),
        max_gross=float(book.gross.max()),
        worst_day=float(d.min()),best_day=float(d.max()),days=len(d))

def old_pair_attribution():
    source=ROOT/'reports/crypto/signal_chain_v1'
    rows=[]
    for phase in ['2025H2_development','2026_exploratory']:
        f=pd.read_parquet(source/f'forecast_rows_{phase}.parquet')
        p=pd.read_parquet(source/'coin_positions_daily.parquet')
        p=p[(p.phase==phase)&(p.strategy=='pair_full_x2_own_sparse_posthoc')]
        w=p.pivot(index='available_at',columns='symbol',values='weight')
        beta=f.pivot(index='available_at',columns='symbol',values='beta_past').reindex(index=w.index,columns=w.columns)
        returns=f.pivot(index='available_at',columns='symbol',values='return24').reindex(index=w.index,columns=w.columns)
        market=f.groupby('available_at').market_return24.first().reindex(w.index) if 'market_return24' in f else returns.mean(axis=1)
        beta_exposure=(w*beta).sum(axis=1)
        gross=(w*np.expm1(returns)).sum(axis=1)
        approximate_market=beta_exposure*market
        rows.append(dict(phase=phase,strategy='pair_full_x2_own_sparse_posthoc',days=len(w),
            average_estimated_beta=float(beta_exposure.mean()),max_abs_estimated_beta=float(beta_exposure.abs().max()),
            mean_price_pnl_bp=float(gross.mean()*1e4),mean_estimated_market_bp=float(approximate_market.mean()*1e4),
            mean_remaining_bp=float((gross-approximate_market).mean()*1e4),
            method='original estimated-beta decomposition; log beta × market log return is an approximation, residual includes beta error and arithmetic convexity'))
    save_csv(pd.DataFrame(rows),'old_pair_attribution.csv')

def policies(pred,price,funding):
    base=pred[pred.model=='state_tree'].set_index('available_at').sort_index()
    clock=base.index
    # Portfolio starts at first legal execution; terminal uses last 5m open.
    price=price.loc[clock[0]+pd.Timedelta('5min'):]
    funding=funding[-len(price):]
    n=len(clock)
    hours=clock.hour.to_numpy()
    rv=base.rv_hour.to_numpy()
    specs={}
    dummy=rv*np.sqrt(24)
    for name,b in [('buy_hold',1.),('fixed_beta_035',.35)]:
        trade=np.zeros(n,dtype=bool);trade[0]=True
        specs[name]=(np.full(n,b),trade,np.full(n,'initial',dtype=object))
    specs['vol_control']=decision_path(dummy,dummy,dummy,dummy,rv,hours,CFG,risk_only=True)
    for name,column in [('trend24','short_signal'),('trend168','slow_signal'),('trend_multi','trend_signal')]:
        score=base[column].to_numpy()
        specs[name]=decision_path(.15*score*dummy,dummy,dummy,dummy,rv,hours,CFG,direct_score=score)
    score=base.trend_signal.to_numpy()
    specs['trend_multi_long_only']=decision_path(.15*score*dummy,dummy,dummy,dummy,rv,hours,CFG,direct_score=score,long_only=True)
    for model,g in pred.groupby('model'):
        if model=='historical_mean':
            continue
        g=g.set_index('available_at').reindex(clock)
        # A fixed linear-drift assumption maps each raw horizon expectation to
        # 24h units. It is NOT an independently calibrated 24h distribution.
        z=np.column_stack([g[f'pred_return{h}_z'] for h in [4,24,72]])
        raw_mu=np.column_stack([g[f'pred_return{h}'] for h in [4,24,72]])
        mu=(raw_mu*np.array([6.,1.,1/3]))@np.array([.2,.5,.3])
        # Latest realized funding is known with >=1h lag. Predictive carry uses
        # three future payments as a conservative fixed assumption; actual
        # future funding remains exclusively in the ledger.
        mu=mu-g.known_funding.to_numpy()/1e4*3
        risk_long=g.pred_long_mae24.to_numpy()
        risk_short=g.pred_short_mae24.to_numpy()
        uncertainty=dummy*(1+np.std(z,axis=1))
        specs[model]=decision_path(mu,risk_long,risk_short,uncertainty,rv,hours,CFG)
        if model in ['state_tree','x2_tree']:
            specs[model+'_long_only']=decision_path(mu,risk_long,risk_short,uncertainty,rv,hours,CFG,long_only=True)
        if model=='state_tree':
            specs['trend_state_risk']=decision_path(.15*score*dummy,risk_long,risk_short,uncertainty,rv,hours,CFG,direct_score=score)
            for review in [2,8]:
                specs[f'state_tree_review{review}h']=decision_path(mu,risk_long,risk_short,uncertainty,rv,hours,CFG,review_hours=review)
            for penalty in [.08,.16]:
                neighbor_cfg={**CFG,'risk_penalty':penalty}
                specs[f'state_tree_penalty{penalty}']=decision_path(mu,risk_long,risk_short,uncertainty,rv,hours,neighbor_cfg)
            # Direction-only / fixed known risk isolates future-risk forecasts.
            specs['state_tree_known_risk']=decision_path(mu,dummy,dummy,uncertainty,rv,hours,CFG)
    summaries=[]; daily_frames=[]; states=[]; attributions=[]; tails=[]; positions=[]; execution_rows=[]; neighborhood=[]
    reference_books={}
    for policy,(target,trade,reason) in specs.items():
        print('Ledger:',policy, 'planned orders:',int(trade.sum()),flush=True)
        order_frame=pd.DataFrame(dict(available_at=clock,target_beta=target,order=trade,reason=reason))
        order_frame['policy']=policy
        execution_rows.append(order_frame)
        for fee in CFG['cost_scenarios_bp_side']:
            book=make_book(price,funding,clock,target,trade,fee)
            book['state']=base.state.reindex(book.index-pd.Timedelta('5min'),method='ffill').to_numpy()
            book['rebound_flag']=base.rebound_flag.reindex(book.index-pd.Timedelta('5min'),method='ffill').fillna(False).to_numpy()
            if not np.allclose(book.net_return,book.price_return-book.fee_return-book.funding_return,rtol=1e-8,atol=2e-14):
                raise AssertionError('cash-flow identity failed')
            for phase,mask in [('all',np.ones(len(book),bool)),('development_2024_2025',book.index.year<2026),('reused_2026',book.index.year>=2026),('recent_2025H2_2026',book.index>=pd.Timestamp('2025-07-01',tz='UTC'))]:
                summaries.append(book_stats(book.loc[mask],policy,fee,phase))
            d=pd.DataFrame(dict(day=book.index.floor('D'),net_return=book.net_return,price_return=book.price_return,
                fee_return=book.fee_return,funding_return=book.funding_return,net_beta=book.net_beta,gross=book.gross,long_price=book.long_price,short_price=book.short_price))
            daily=d.groupby('day').agg(net_return=('net_return',lambda s:(1+s).prod()-1),price_bp=('price_return',lambda s:s.sum()*1e4),fee_bp=('fee_return',lambda s:s.sum()*1e4),funding_bp=('funding_return',lambda s:s.sum()*1e4),net_beta=('net_beta','mean'),gross=('gross','mean'),long_bp=('long_price',lambda s:s.sum()*1e4),short_bp=('short_price',lambda s:s.sum()*1e4)).reset_index()
            daily['policy'],daily['fee_bp_side']=policy,fee
            daily_frames.append(daily)
            if fee==6:
                reference_books[policy]=book
                dump=book.reset_index(names='execution_at')
                dump['policy']=policy
                positions.append(dump)
                mr=np.diff(price.to_numpy(),axis=0)/price.to_numpy()[:-1]
                market=mr@np.array(CFG['market_weights'])
                b=book.net_beta.to_numpy()
                for phase,mask in [('all',np.ones(len(book),bool)),('development',book.index.year<2026),('reused_2026',book.index.year>=2026)]:
                    exposure=b[mask].mean()*market[mask].mean()
                    timing=np.mean((b[mask]-b[mask].mean())*(market[mask]-market[mask].mean()))
                    tilt=book.price_return.to_numpy()[mask].mean()-np.mean(b[mask]*market[mask])
                    attributions.append(dict(policy=policy,phase=phase,price_mean_bp_5m=float(book.price_return.to_numpy()[mask].mean()*1e4),average_beta_component_bp=exposure*1e4,timing_component_bp=timing*1e4,drift_tilt_bp=tilt*1e4,fees_mean_bp=float(book.fee_return.to_numpy()[mask].mean()*1e4),funding_mean_bp=float(book.funding_return.to_numpy()[mask].mean()*1e4),net_mean_bp=float(book.net_return.to_numpy()[mask].mean()*1e4)))
                for state,g in book.groupby('state'):
                    states.append(dict(policy=policy,state=state,hours=len(g)/12,price_bp_hour=float(g.price_return.mean()*12e4),net_bp_hour=float(g.net_return.mean()*12e4),participation=float((g.gross>.05).mean()),short_time=float((g.net_beta<-.05).mean()),short_price_bp_sum=float(g.short_price.sum()*1e4)))
                for year,g in book.groupby(book.index.year):
                    for event,mask in [('all',np.ones(len(g),bool)),('rebound',g.rebound_flag.to_numpy()),('past_downtrend',(g.state=='downtrend').to_numpy())]:
                        if mask.any():
                            states.append(dict(policy=policy,state=f'{year}_{event}',hours=mask.sum()/12,price_bp_hour=float(g.loc[mask,'price_return'].mean()*12e4),net_bp_hour=float(g.loc[mask,'net_return'].mean()*12e4),participation=float((g.loc[mask,'gross']>.05).mean()),short_time=float((g.loc[mask,'net_beta']<-.05).mean()),short_price_bp_sum=float(g.loc[mask,'short_price'].sum()*1e4)))
                values=daily.net_return.to_numpy()
                for k in [1,5,10]:
                    tops=np.sort(values)[-k:]
                    tails.append(dict(policy=policy,best_days_deleted=k,remaining_return=float(np.expm1(np.log1p(values).sum()-np.log1p(tops).sum()))))
    # Delay perturbation has no re-estimation or action changes.
    for policy in ['trend_multi','state_tree','x2_tree']:
        target,trade,_=specs[policy]
        for delay in [10,20]:
            book=make_book(price,funding,clock,target,trade,6,delay)
            neighborhood.append(book_stats(book,policy+f'_delay{delay}m',6,'all'))
    save_csv(pd.DataFrame(summaries),'policy_scores.csv')
    daily=pd.concat(daily_frames,ignore_index=True)
    save_csv(daily,'portfolio_daily.csv')
    save_csv(pd.DataFrame(states),'state_failure_cards.csv')
    save_csv(pd.DataFrame(attributions),'beta_attribution.csv')
    save_csv(pd.DataFrame(tails),'best_day_deletion.csv')
    save_csv(pd.DataFrame(neighborhood),'execution_neighborhood.csv')
    pd.concat(positions,ignore_index=True).to_parquet(DATA/'positions_5m.parquet',index=False)
    pd.concat(execution_rows,ignore_index=True).to_parquet(DATA/'decision_rows.parquet',index=False)
    comparisons=[]
    dp=daily[daily.fee_bp_side==6].pivot(index='day',columns='policy',values='net_return')
    for candidate,reference in [('trend168','vol_control'),('state_ridge','trend168'),('state_tree','trend_multi'),('x2_tree','state_tree'),('x2_soft_local','x2_tree'),('state_tree','state_tree_long_only'),('trend_multi','trend_multi_long_only'),('x2_no_internal','x2_tree'),('state_tree','state_tree_known_risk')]:
        for phase,mask in [('development',dp.index.year<2026),('reused_2026',dp.index.year>=2026),('all',np.ones(len(dp),bool))]:
            mean,low,high=block_interval((dp[candidate]-dp[reference]).loc[mask],CFG['bootstrap_block_days'],CFG['bootstrap_repetitions'])
            comparisons.append(dict(candidate=candidate,reference=reference,phase=phase,mean_increment_bp_day=mean*1e4,low_bp=low*1e4,high_bp=high*1e4))
    save_csv(pd.DataFrame(comparisons),'policy_uncertainty.csv')
    # Hindsight exposure matching is an attribution ruler, never deployable.
    matching=[]
    for policy in ['trend168','trend_multi','state_ridge','state_tree','x2_tree','x2_soft_local']:
        b=reference_books[policy].net_beta.mean()
        trade=hours%4==0;trade[0]=True
        lower,upper=-.75,1.
        for _ in range(24):
            constant=(lower+upper)/2
            matched=make_book(price,funding,clock,np.full(n,constant),trade,6)
            if matched.net_beta.mean()>b:
                upper=constant
            else:
                lower=constant
        stat=book_stats(matched,'expost_matched_'+policy,6,'all')
        stat['matched_mean_beta']=b
        stat['mean_beta_error']=float(matched.net_beta.mean()-b)
        stat['constant_target']=constant
        stat['purpose']='hindsight realized mean net Beta match; 4h constant-target maintenance with paid drift turnover; not deployable calibration'
        matching.append(stat)
    save_csv(pd.DataFrame(matching),'matched_exposure_diagnostics.csv')
    # Proxy stress: same frozen trend definition, BTC/ETH main inputs. These
    # isolate traded proxy, not independently trained proxy-specific forecasts.
    proxies=[]
    for name,w in [('BTC',[1.,0.]),('ETH',[0.,1.])]:
        for policy in ['buy_hold','trend168','trend_multi','state_ridge','state_tree']:
            target,trade,_=specs[policy]
            book=make_book(price,funding,clock,target,trade,6,weights=w)
            stat=book_stats(book,policy,6,'all');stat['proxy']=name
            stat['interpretation']='same joint-market signal; proxy risk target not refitted'
            proxies.append(stat)
    save_csv(pd.DataFrame(proxies),'proxy_sensitivity.csv')
    # Twelve-coin proxy is price-only: missing funding must not become a claim
    # of zero funding. Main two-coin no-funding comparison uses the same clock.
    twelve={}
    for s in CFG['context_symbols']:
        f=pd.read_parquet(Path(CFG['raw_root'])/s/'5m.parquet',columns=['open_time','open'])
        f.index=pd.to_datetime(f.open_time,unit='ms',utc=True)
        twelve[s]=f.open.reindex(price.index)
    p12=pd.DataFrame(twelve)
    proxy_price_rows=[]
    for proxy,p,w in [('BTC_ETH_price_only',price,np.array([.5,.5])),('EW12_selected_panel_price_only',p12,np.ones(12)/12)]:
        for policy in ['buy_hold','trend168','trend_multi','state_ridge','state_tree']:
            target,trade,_=specs[policy]
            target5,orders5=expand_orders(p.index,clock,target,trade)
            values=quantity_ledger(p.to_numpy(),target5,orders5,np.zeros(p.shape),w,6)
            book=pd.DataFrame(values,index=p.index[:-1],columns=BOOK_COLUMNS+[f'weight_{s}' for s in p.columns])
            stat=book_stats(book,policy,6,'all')
            stat['proxy'],stat['funding_status']=proxy,'excluded; price-only proxy comparison'
            proxy_price_rows.append(stat)
    save_csv(pd.DataFrame(proxy_price_rows),'proxy_price_only.csv')
    return pd.DataFrame(summaries),reference_books

def main():
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    hourly,five,price,funding,known,raw_manifest=load_inputs()
    features,rv=market_features(hourly,CFG['market_symbols'],CFG['market_weights'],CFG['context_symbols'],known)
    # All causal state transformations must reproduce at historical prefixes.
    prefix_checks=[]
    for count in [2500,12000,25000]:
        sub={s:f.iloc[:count] for s,f in hourly.items()}
        f2,r2=market_features(sub,CFG['market_symbols'],CFG['market_weights'],CFG['context_symbols'],known)
        error=float(np.nanmax(np.abs(f2.to_numpy()-features.iloc[:count].to_numpy())))
        if error>1e-10:
            raise AssertionError('feature prefix changed')
        prefix_checks.append(dict(hours=count,max_abs_error=error))
    feature_registry(CFG['path_lags_hours']).to_csv(OUT/'feature_registry.csv',index=False,encoding='utf-8-sig')
    features.to_parquet(DATA/'market_features.parquet')
    corr=features[features.index<pd.Timestamp('2024-01-01',tz='UTC')].corr()
    corr.to_csv(OUT/'initial_feature_correlation.csv',encoding='utf-8-sig')
    pred,x=forecasts(features,rv,price)
    prediction_scores(pred)
    old_pair_attribution()
    summaries,books=policies(pred,price,funding)
    # Source-only changes, recorded before/after execution, must be absent.
    for r in raw_manifest:
        path=Path(r['path'])
        if path.stat().st_mtime_ns!=r['mtime_ns'] or path.stat().st_size!=r['bytes']:
            raise AssertionError('raw source modified')
    manifest=dict(research_id=CFG['research_id'],frozen_on='2026-10-04',historical_status=CFG['historical_status'],
        config=CFG,config_sha256=hashlib.sha256((ROOT/'configs/crypto_market_beta_a.json').read_bytes()).hexdigest(),
        raw_sources=raw_manifest,prefix_checks=prefix_checks,python=platform.python_version(),
        pandas=pd.__version__,numpy=np.__version__,lightgbm=lgb.__version__,
        model_families=MODELS,policy_count=int(summaries.policy.nunique()),
        future_confirmation_completed=False,live_orders_sent=False,
        funding_price='official settled rate times 5m trade-open mark proxy; not exact exchange cashflow',
        account_fee_verified=False)
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'config_snapshot.json').write_text(json.dumps(CFG,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summaries[(summaries.phase=='all')&(summaries.fee_bp_side==6)][['policy','return_total','max_drawdown','mean_net_beta','participation','mean_sign_episode_hours']].to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--scores-only',action='store_true',help='refresh prediction diagnostics without fitting or changing policies')
    args=parser.parse_args()
    if args.scores_only:
        prediction_scores(pd.read_parquet(DATA/'forward_predictions.parquet'))
    else:
        main()
