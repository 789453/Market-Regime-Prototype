"""12-contract strategy research: economic experts, drift adaptation, direct utility."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.crypto.beta_strategy import (EXPERTS,REGIMES,coin_features,economic_regime,expert_signals,gate_weights,
    rebalance_targets,adaptive_experts,markov_filter,kalman_drift,feature_registry)
from src.crypto.market_beta import quantity_ledger,BOOK_COLUMNS
CFG=json.loads((ROOT/'configs/crypto_beta_strategy_v2.json').read_text())
OUT=ROOT/CFG['output_dir'];DATA=ROOT/CFG['data_dir']
PHASES={'all':None,'development_2024_25':(2024,2025),'reused_2026':(2026,), 'year2024':(2024,), 'year2025':(2025,)}

def csv(f,name):
    if name=='contract_daily.csv':
        f.to_csv(OUT/(name+'.gz'),index=False,encoding='utf-8-sig',compression='gzip')
    else:f.to_csv(OUT/name,index=False,encoding='utf-8-sig')
def say(s):print(s,flush=True)

def load_data():
    hourly={};prices={};manifest=[]
    end=pd.Timestamp(CFG['end_exclusive'])
    for s in CFG['symbols']:
        for freq in ['1h','5m']:
            path=Path(CFG['raw_root'])/s/f'{freq}.parquet'
            cols=['open_time','close','quote_volume','taker_buy_quote_volume'] if freq=='1h' else ['open_time','open']
            f=pd.read_parquet(path,columns=cols)
            f.index=pd.DatetimeIndex(pd.to_datetime(f.open_time,unit='ms',utc=True)).as_unit('ns')
            if freq=='1h':f.index+=pd.Timedelta('1h')
            f=f.loc[f.index<=end] if freq=='1h' else f.loc[f.index<end]
            if f.index.has_duplicates or not np.all(np.diff(f.index.asi8)==pd.Timedelta(freq).value):raise ValueError(f'grid {s} {freq}')
            if freq=='1h':hourly[s]=f
            else:prices[s]=f.open
            manifest.append(dict(path=str(path),rows=len(f),first=str(f.index[0]),last=str(f.index[-1]),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    p=pd.DataFrame(prices)
    r=np.log(pd.DataFrame({s:f.close for s,f in hourly.items()})).diff()
    market=(r.BTCUSDT+r.ETHUSDT)/2
    breadth=(r.rolling(24).sum()>0).mean(axis=1)
    features={s:coin_features(hourly[s],market,breadth) for s in CFG['symbols']}
    return hourly,p,features,manifest

def entry_labels(f,price):
    pos=price.index.get_indexer(f.index+pd.Timedelta('5min'))
    out=pd.DataFrame(index=f.index)
    for h in [1,4,24,72,240]:
        e=pos+h*12;valid=(pos>=0)&(e<len(price))
        y=np.full(len(f),np.nan);y[valid]=price.to_numpy()[e[valid]]/price.to_numpy()[pos[valid]]-1
        out[f'r{h}']=y
    return out

def folds(index):
    starts=pd.date_range('2024-01-01','2026-07-01',freq='QS',tz='UTC')
    for j,start in enumerate(starts):
        finish=starts[j+1] if j+1<len(starts) else pd.Timestamp(CFG['end_exclusive'])
        yield start,finish

def rolling_choice(f,experts,labels,cap,kind,rows):
    # Matured price returns of fixed one-hour target proxy; actual results use 5m quantity ledger.
    cols=['fast','medium','slow'] if kind=='trend' else ['break48','break168','break336']
    signals=experts[cols].to_numpy();ret=labels.r1.to_numpy()
    payoff=signals*cap[:,None]*ret[:,None]-2e-4*np.abs(np.diff(signals*cap[:,None],axis=0,prepend=0))
    out=np.zeros(len(f))
    for start,end in folds(f.index):
        tr=(f.index>=start-pd.Timedelta('365D'))&(f.index+pd.Timedelta('1h5min')<start)
        te=(f.index>=start)&(f.index<end)
        x=payoff[tr]
        # Discounted historical average / RMS with shrinkage to equal selection prior.
        w=np.exp(-np.log(2)*np.arange(len(x)-1,-1,-1)/(126*24))
        m=np.average(x,axis=0,weights=w);rms=np.sqrt(np.average(x*x,axis=0,weights=w)).clip(1e-6)
        score=m/rms;best=int(np.argmax(score))
        out[te]=signals[te,best]
        rows.append(dict(symbol=f.attrs['symbol'],fold=str(start),kind=kind,selected=cols[best],scores=json.dumps(score.tolist()),train_end=str(f.index[tr][-1]+pd.Timedelta('1h5min'))))
    return out

def train_filtered_moe(f,experts,labels,rows):
    columns=['trend72','trend168','eff72','risk_ratio','recovery24']
    x=f[columns].to_numpy();e=experts[EXPERTS].to_numpy()
    state=economic_regime(f);prior=gate_weights(state)
    out=np.zeros(len(f));latent=np.full(len(f),-1)
    for start,end in folds(f.index):
        tr=(f.index>=start-pd.Timedelta('365D'))&(f.index+pd.Timedelta('24h5min')<start)
        te=(f.index>=start)&(f.index<end)
        scaler=StandardScaler().fit(x[tr]);z=np.clip(scaler.transform(x),-5,5)
        gmm=GaussianMixture(4,covariance_type='diag',reg_covar=.10,max_iter=60,random_state=CFG['seed']).fit(z[tr][::4])
        emission=gmm.predict_proba(z[tr])
        transition=emission[:-1].T@emission[1:]+50*np.eye(4)+1
        transition/=transition.sum(axis=1,keepdims=True)
        trainq=markov_filter(emission/gmm.weights_[None,:],transition,gmm.weights_)
        reward=e[tr]*np.clip(labels.r24.to_numpy()[tr]/(f.rv.to_numpy()[tr]*np.sqrt(24)),-4,4)[:,None]
        mass=trainq.sum(axis=0)
        actionvalue=trainq.T@reward/(mass[:,None]+720)
        softprior=trainq.T@prior[tr]/mass[:,None]
        specialist=softmax(3*actionvalue+np.log(.85*softprior+.15/len(EXPERTS)),axis=1)
        bridge=(f.index>f.index[tr][-1])&(f.index<start)
        initial=markov_filter(gmm.predict_proba(z[bridge])/gmm.weights_[None,:],transition,trainq[-1])[-1] if bridge.any() else trainq[-1]
        q=markov_filter(gmm.predict_proba(z[te])/gmm.weights_[None,:],transition,initial)
        weights=q@specialist
        out[te]=np.sum(weights*e[te],axis=1);latent[te]=q.argmax(axis=1)
        rows.append(dict(symbol=f.attrs['symbol'],fold=str(start),method='filtered_moe',train_n=int(tr.sum()),train_label_end=str(f.index[tr][-1]+pd.Timedelta('24h5min')),test_start=str(start),latent_means=json.dumps(gmm.means_.tolist()),transition=json.dumps(transition.tolist()),specialist_weights=json.dumps(specialist.tolist())))
    return out,latent

class DirectTCN(torch.nn.Module):
    def __init__(self,d):
        super().__init__();self.conv=torch.nn.Conv1d(d,16,3);self.head=torch.nn.Sequential(torch.nn.Linear(16,16),torch.nn.Tanh(),torch.nn.Linear(16,1))
    def forward(self,x):
        z=torch.tanh(self.conv(torch.nn.functional.pad(x.transpose(1,2),(2,0))))[:,:,-1]
        return torch.tanh(self.head(z)).squeeze(-1)

def pooled_models(features,labels,rows):
    columns=[c for c in next(iter(features.values())).columns if c!='rv']
    # A temporal sequence encodes completed snapshots t-42,...,t at 6h spacing.
    frames=[]
    for s,f in features.items():
        row=f[columns].copy();row['symbol']=s;row['t']=f.index
        for h in [4,24,72,240]:row[f'y{h}']=labels[s][f'r{h}']/(f.rv*np.sqrt(h))
        row['cap']=np.minimum(1.,CFG['vol_target']/(f.rv*np.sqrt(8760)))
        row['r4']=labels[s].r4
        frames.append(row)
    panel=pd.concat(frames,ignore_index=True)
    # Coin identity deliberately omitted: shared economic relationships, local scales.
    result={s:{m:np.zeros(len(f)) for m in ['expanding_tree','rolling_tree','action_tcn']} for s,f in features.items()}
    array=panel[columns].to_numpy(dtype=np.float32)
    seq=np.stack([pd.concat([f[columns].shift(l) for f in features.values()],ignore_index=True).to_numpy(dtype=np.float32) for l in [42,36,30,24,18,12,6,0]],axis=1)
    times=pd.DatetimeIndex(panel.t)
    torch.set_num_threads(4)
    for fold,(start,end) in enumerate(folds(next(iter(features.values())).index)):
        eligible=(times+pd.Timedelta('240h5min')<start)&np.isfinite(panel[[f'y{h}' for h in [24,72,240]]]).all(axis=1).to_numpy()
        test=(times>=start)&(times<end)
        sparse=(np.arange(len(panel))%4==0)
        for method in ['expanding_tree','rolling_tree']:
            train=eligible&sparse
            if method=='rolling_tree':train&=(times>=start-pd.Timedelta('365D'))
            x=array[train];xt=array[test];pred=[]
            for h in [24,72,240]:
                model=lgb.LGBMRegressor(n_estimators=120,num_leaves=15,max_depth=5,min_child_samples=480,learning_rate=.035,reg_lambda=30,verbosity=-1,n_jobs=4,random_state=CFG['seed'])
                decay=np.exp(-np.log(2)*((start-times[train]).total_seconds()/86400)/(126 if method=='rolling_tree' else 100000))
                model.fit(x,np.clip(panel[f'y{h}'].to_numpy()[train],-5,5),sample_weight=decay)
                pred.append(model.booster_.predict(xt,num_threads=4))
            score=np.tanh((.45*pred[0]+.35*pred[1]+.2*pred[2])/.22)
            for s in CFG['symbols']:
                mask=panel.symbol.to_numpy()[test]==s
                local=(features[s].index>=start)&(features[s].index<end)
                result[s][method][local]=score[mask]
            rows.append(dict(symbol='POOL12',fold=str(start),method=method,train_n=int(train.sum()),train_label_end=str(times[train].max()+pd.Timedelta('240h5min')),test_start=str(start)))
        # Direct action learning uses nonoverlapping 4h targets, with only 4h maturity.
        train=(times>=start-pd.Timedelta('365D'))&(times+pd.Timedelta('4h5min')<start)&sparse&np.isfinite(seq).all(axis=(1,2))&panel.r4.notna().to_numpy()
        scaler=StandardScaler().fit(array[train])
        xs=np.clip((seq[train]-scaler.mean_)/scaler.scale_,-5,5).astype(np.float32)
        # Adjacent 4h snapshots penalize excessive direction/size changes within each coin.
        prev=np.empty_like(seq);prev[:]=np.nan
        previous_cap=np.zeros(len(panel))
        for j,s in enumerate(CFG['symbols']):
            a=j*len(features[s]);b=a+len(features[s]);prev[a+4:b]=seq[a:b-4]
            previous_cap[a+4:b]=panel.cap.to_numpy()[a:b-4]
        xp=np.nan_to_num(np.clip((prev[train]-scaler.mean_)/scaler.scale_,-5,5)).astype(np.float32)
        torch.manual_seed(CFG['seed']+fold)
        network=DirectTCN(len(columns));optimizer=torch.optim.AdamW(network.parameters(),lr=.001,weight_decay=.01)
        inputs=torch.tensor(xs);before=torch.tensor(xp)
        returns=torch.tensor(panel.r4.to_numpy()[train],dtype=torch.float32)
        caps=torch.tensor(panel.cap.to_numpy()[train],dtype=torch.float32)
        previous_caps=torch.tensor(previous_cap[train],dtype=torch.float32)
        rng=np.random.default_rng(CFG['seed']+fold)
        for epoch in range(10):
            for batch in np.array_split(rng.permutation(len(xs)),max(1,int(np.ceil(len(xs)/2048)))):
                q=network(inputs[batch])*caps[batch]
                qprev=network(before[batch])*previous_caps[batch]
                pnl=q*returns[batch]-2e-4*torch.abs(q-qprev)
                # Stable utility surrogate, not a reproduction of original DMN Sharpe loss.
                loss=-100*pnl.mean()+150*(pnl**2).mean()+.003*(q*q).mean()
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),1);optimizer.step()
        xt=np.nan_to_num(np.clip((seq[test]-scaler.mean_)/scaler.scale_,-5,5)).astype(np.float32)
        with torch.no_grad():score=np.concatenate([network(torch.tensor(a)).numpy() for a in np.array_split(xt,max(1,int(np.ceil(len(xt)/4096))))])
        for s in CFG['symbols']:
            mask=panel.symbol.to_numpy()[test]==s
            local=(features[s].index>=start)&(features[s].index<end)
            result[s]['action_tcn'][local]=score[mask]
        rows.append(dict(symbol='POOL12',fold=str(start),method='action_tcn',train_n=int(train.sum()),train_label_end=str(times[train].max()+pd.Timedelta('4h5min')),test_start=str(start),parameters=sum(p.numel() for p in network.parameters())))
        joblib.dump(dict(columns=columns,scaler=scaler,state_dict={k:v.numpy() for k,v in network.state_dict().items()},fold=str(start)),OUT/f'tcn_{start.date()}.joblib',compress=3)
        say(f'Pooled models {start.date()} done')
    return result

def build_signals(features,labels,hourly):
    choices=[];fits=[];all_signals={};state_rows=[];weights_rows=[]
    for s,f in features.items():
        f.attrs['symbol']=s
        e=expert_signals(f);p=np.log(hourly[s].close.reindex(f.index))
        for h in [48,168,336]:
            lo=p.shift(1).rolling(h).min();hi=p.shift(1).rolling(h).max()
            e[f'break{h}']=np.clip(2*(p-lo)/(hi-lo)-1,-1,1)
        # Use actual channel geometry for breakout specialist, not another momentum alias.
        e['breakout']=(e.break48+e.break168)/2
        state=economic_regime(f);prior=gate_weights(state)
        cap=np.minimum(1.,CFG['vol_target']/(f.rv.to_numpy()*np.sqrt(8760)))
        signals={f'trend{h}':np.tanh(f[f'trend{h}'].to_numpy()/.8) for h in [24,72,168,336]}
        signals.update(trend_ensemble=e[['fast','medium','slow']].mean(axis=1).to_numpy(),breakout=e.breakout.to_numpy(),range_only=e['range'].to_numpy(),down_rebound=e.down_rebound.to_numpy())
        signals['selected_trend']=rolling_choice(f,e,labels[s],cap,'trend',choices)
        signals['selected_breakout']=rolling_choice(f,e,labels[s],cap,'breakout',choices)
        signals['regime_gate']=(prior*e[EXPERTS].to_numpy()).sum(axis=1)
        signals['gate_no_range']=(np.where(np.arange(7)==4,0,prior)*e[EXPERTS].to_numpy()).sum(axis=1)
        ep=e[EXPERTS].to_numpy();lagreward=np.zeros_like(ep)
        # Entry t+5m -> t+1h5m; only available at completed hour t+2h.
        lagreward[2:]=np.clip(ep[:-2]*labels[s].r1.to_numpy()[:-2,None]/(f.rv.to_numpy()[:-2,None]*np.sqrt(24)),-1,1)
        lagreward[2:]-=2e-4*np.abs(np.diff(ep[:-2]*cap[:-2,None],axis=0,prepend=0))/(f.rv.to_numpy()[:-2,None]*np.sqrt(24))
        for hl in [14,42,126]:
            q,w=adaptive_experts(ep,state,prior,np.nan_to_num(lagreward),hl)
            signals[f'adaptive{hl}']=q
            if hl==42:
                sample=pd.DataFrame(w[::24],columns=EXPERTS);sample['symbol']=s;sample['available_at']=f.index[::24];sample['state']=[REGIMES[i] for i in state[::24]];weights_rows.append(sample)
        signals['filtered_moe'],latent=train_filtered_moe(f,e,labels[s],fits)
        signals['kalman']=kalman_drift(np.log(hourly[s].close).diff().reindex(f.index).fillna(0).to_numpy(),f.rv.to_numpy())
        signals['trend_ensemble_long']=np.maximum(0,signals['trend_ensemble'])
        signals['regime_gate_long']=np.maximum(0,signals['regime_gate'])
        all_signals[s]=signals
        row=pd.DataFrame(dict(available_at=f.index,symbol=s,state=[REGIMES[i] for i in state],latent=latent,market_beta=f.market_beta,rv=f.rv,risk_ratio=f.risk_ratio,trend168=f.trend168));state_rows.append(row)
        say(f'Economic / adaptive / filtered strategies {s} done')
    supervised=pooled_models(features,labels,fits)
    for s in CFG['symbols']:all_signals[s].update(supervised[s])
    csv(pd.DataFrame(choices),'parameter_choices.csv');csv(pd.DataFrame(fits),'fit_audit.csv');csv(pd.concat(weights_rows),'adaptive_weights_daily.csv')
    pd.concat(state_rows).to_parquet(DATA/'states.parquet',index=False)
    return all_signals

def metrics(day,policy,symbol,fee,phase):
    eq=(1+day.net_return).cumprod();dd=eq/eq.cummax().clip(lower=1)-1
    vol=day.net_return.std()*np.sqrt(365)
    return dict(policy=policy,symbol=symbol,fee_bp=fee,phase=phase,days=len(day),return_total=float(eq.iloc[-1]-1),sharpe=float(day.net_return.mean()*365/vol) if vol>0 else 0.,max_dd=float(dd.min()),vol=float(vol),long_time=float(day.long_time.mean()),short_time=float(day.short_time.mean()),gross=float(day.gross.mean()),turnover=float(day.turnover.sum()),long_price_sum=float(day.long_price.sum()),short_price_sum=float(day.short_price.sum()),fee_sum=float(day.fee_return.sum()))

def run_backtests(features,prices,signals):
    stats=[];daily=[];episodes=[];attribution=[];statecards=[];hourout=[];fundsens=[]
    start=pd.Timestamp(CFG['first_forward']);price=prices.loc[prices.index>=start+pd.Timedelta('5min')]
    all_index=price.index[:-1];market_returns=((price.BTCUSDT.pct_change().shift(-1)+price.ETHUSDT.pct_change().shift(-1))/2).iloc[:-1].to_numpy()
    existing=pd.read_parquet(ROOT/'data/crypto/market_beta_a_v1/official_funding.parquet')
    for s,f in features.items():
        clock=f.index[(f.index>=start)&(f.index+pd.Timedelta('5min')<price.index[-1])]
        local=f.reindex(clock)
        pos=all_index.get_indexer(clock+pd.Timedelta('5min'))
        regimes=economic_regime(local)
        fullreg=pd.Series(regimes,index=clock+pd.Timedelta('5min')).reindex(all_index,method='ffill').to_numpy()
        beta=pd.Series(local.market_beta.to_numpy(),index=clock+pd.Timedelta('5min')).reindex(all_index,method='ffill').to_numpy()
        own=price[[s]].to_numpy();zero_fund=np.zeros_like(own)
        strategies=dict(signals[s]);strategies['vol_long']=np.ones(len(f));strategies['hold']=np.ones(len(f))
        for name,raw in strategies.items():
            desired=pd.Series(raw,index=f.index).reindex(clock).to_numpy()
            cap=np.minimum(1.,CFG['vol_target']/(local.rv.to_numpy()*np.sqrt(8760)))
            desired=np.clip(desired,-1,1)*cap
            if name=='hold':desired=np.ones(len(clock))
            target,order=rebalance_targets(desired,clock.hour.to_numpy(),CFG['rebalance_hours'],CFG['min_resize'])
            if name=='hold':target[:]=1;order[:]=False;order[0]=True
            t5=np.zeros(len(all_index));o5=np.zeros(len(all_index),dtype=np.bool_);t5[pos]=target;o5[pos]=order
            if name!='hold':
                sign=np.sign(target);edges=np.flatnonzero(np.r_[True,sign[1:]!=sign[:-1]])
                for j,a in enumerate(edges):
                    b=edges[j+1] if j+1<len(edges) else len(target)
                    if sign[a]:episodes.append(dict(symbol=s,policy=name,start=str(clock[a]),hours=b-a,sign=sign[a],right_censored=b==len(target)))
            for fee in [2,5]:
                values=quantity_ledger(own,t5,o5,zero_fund,np.ones(1),fee)
                book=pd.DataFrame(values,index=all_index,columns=BOOK_COLUMNS+['weight'])
                book['long_time']=(book.net_beta>.03).astype(float);book['short_time']=(book.net_beta<-.03).astype(float)
                d=book.resample('D').agg(net_return=('net_return',lambda x:(1+x).prod()-1),equity=('equity','last'),long_time=('long_time','mean'),short_time=('short_time','mean'),gross=('gross','mean'),turnover=('turnover','sum'),long_price=('long_price','sum'),short_price=('short_price','sum'),fee_return=('fee_return','sum'))
                d['symbol']=s;d['policy']=name;d['fee_bp']=fee;d.index.name='day';daily.append(d.reset_index())
                for phase,years in PHASES.items():
                    g=d if years is None else d.loc[d.index.year.isin(years)]
                    if len(g):stats.append(metrics(g,name,s,fee,phase))
                if fee==2:
                    effective=book.net_beta.to_numpy()*beta
                    r=book.price_return.to_numpy();factor=effective*market_returns;residual=r-factor
                    for phase,years in PHASES.items():
                        mask=np.ones(len(book),bool) if years is None else book.index.year.isin(years)
                        b=effective[mask];m=market_returns[mask]
                        avg=b.mean()*m.mean();cov=np.mean((b-b.mean())*(m-m.mean()))
                        attribution.append(dict(symbol=s,policy=name,phase=phase,periods=int(mask.sum()),mean_factor_beta=float(b.mean()),average_exposure_sum=float(avg*mask.sum()),timing_covariance_sum=float(cov*mask.sum()),residual_sum=float(residual[mask].sum()),price_sum=float(r[mask].sum()),fee_sum=float(book.fee_return.to_numpy()[mask].sum()),identity_error=float(abs((avg+cov)*mask.sum()+residual[mask].sum()-r[mask].sum()))))
                    for regime in range(6):
                        mask=fullreg==regime
                        statecards.append(dict(symbol=s,policy=name,state=REGIMES[regime],hours=float(mask.sum()/12),price_bp_hour=float(book.price_return.to_numpy()[mask].sum()/max(mask.sum()/12,1)*1e4),long_bp_hour=float(book.long_price.to_numpy()[mask].sum()/max(mask.sum()/12,1)*1e4),short_bp_hour=float(book.short_price.to_numpy()[mask].sum()/max(mask.sum()/12,1)*1e4),fee_bp_hour=float(book.fee_return.to_numpy()[mask].sum()/max(mask.sum()/12,1)*1e4)))
                    # Local records retain hourly exposure and actual next price outcomes.
                    h=book.resample('h').agg(net_return=('net_return',lambda x:(1+x).prod()-1),net_beta=('net_beta','mean'),long_price=('long_price','sum'),short_price=('short_price','sum'))
                    h['symbol']=s;h['policy']=name;hourout.append(h.reset_index(names='hour'))
                    if s in ['BTCUSDT','ETHUSDT']:
                        funding=np.zeros_like(own);events=existing.loc[existing.symbol==s]
                        epos=price.index.searchsorted(pd.DatetimeIndex(events.settlement_at).as_unit('ns'));valid=epos<len(price)
                        np.add.at(funding[:,0],epos[valid],events.last_funding_rate.to_numpy()[valid])
                        funded=quantity_ledger(own,t5,o5,funding,np.ones(1),fee)
                        fundsens.append(dict(symbol=s,policy=name,fee_bp=fee,price_fee_return=float(book.equity.iloc[-1]-1),with_local_funding_return=float(funded[-1,5]-1),funding_sum=float(funded[:,3].sum())))
            say(f'Backtest {s} {name}')
    daily=pd.concat(daily,ignore_index=True)
    portfolios=[]
    for (name,fee),g in daily.groupby(['policy','fee_bp']):
        # Initial 1/12 independent sleeves. Their equity shares drift; no free rebalancing.
        eq=g.pivot(index='day',columns='symbol',values='equity')
        if eq.isna().any().any():raise ValueError('incomplete universe')
        wealth=eq.mean(axis=1);returns=wealth.pct_change();returns.iloc[0]=wealth.iloc[0]-1
        opening=eq.shift(1).fillna(1);w=opening.div(opening.sum(axis=1),axis=0)
        d=pd.DataFrame(dict(net_return=returns,equity=wealth))
        for col in ['long_time','short_time','gross','turnover','long_price','short_price','fee_return']:
            d[col]=(g.pivot(index='day',columns='symbol',values=col)*w).sum(axis=1)
        d['symbol']='PORT12';d['policy']=name;d['fee_bp']=fee;d.index.name='day';portfolios.append(d.reset_index())
        for phase,years in PHASES.items():
            block=d if years is None else d.loc[d.index.year.isin(years)]
            if len(block):stats.append(metrics(block,name,'PORT12',fee,phase))
    csv(pd.DataFrame(stats),'strategy_scores.csv');csv(pd.concat(portfolios),'portfolio_daily.csv');csv(daily,'contract_daily.csv')
    csv(pd.DataFrame(episodes),'episodes.csv');csv(pd.DataFrame(attribution),'timing_attribution.csv');csv(pd.DataFrame(statecards),'state_strategy_cards.csv');csv(pd.DataFrame(fundsens),'local_funding_sensitivity.csv')
    pd.concat(hourout).to_parquet(DATA/'hourly_positions.parquet',index=False)
    return daily

def main():
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    began=time.time();hourly,price,features,manifest=load_data()
    # Common warmup and complete clocks; local raw files remain untouched.
    for s,f in features.items():features[s]=f.loc[f.notna().all(axis=1)].copy()
    clock=next(iter(features.values())).index
    for f in features.values():clock=clock.intersection(f.index)
    features={s:f.reindex(clock) for s,f in features.items()}
    labels={s:entry_labels(f,price[s]) for s,f in features.items()}
    pd.concat([f.reset_index(names='available_at').assign(symbol=s) for s,f in features.items()]).to_parquet(DATA/'features.parquet',index=False)
    csv(feature_registry(),'feature_registry.csv')
    signals=build_signals(features,labels,hourly)
    pd.concat([pd.DataFrame(v,index=features[s].index).reset_index(names='available_at').assign(symbol=s) for s,v in signals.items()]).to_parquet(DATA/'signals.parquet',index=False)
    run_backtests(features,price,signals)
    audit=pd.read_csv(OUT/'fit_audit.csv')
    maturity=all(pd.Timestamp(r.train_label_end)<pd.Timestamp(r.test_start) for r in audit.itertuples())
    (OUT/'run_manifest.json').write_text(json.dumps(dict(config=CFG,raw_files=manifest,policies=list(next(iter(signals.values())))+['vol_long','hold'],folds=11,all_train_labels_matured=maturity,elapsed_seconds=time.time()-began,torch_version=torch.__version__,primary_funding='excluded',status='historical walk-forward, not untouched OOS'),ensure_ascii=False,indent=2),encoding='utf-8')
    say(f'FINISHED {time.time()-began:.1f}s; maturity={maturity}')

if __name__=='__main__':main()
