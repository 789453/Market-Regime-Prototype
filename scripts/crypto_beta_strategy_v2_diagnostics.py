"""Research interpretation: exact leg contributions, state transfers, covariance."""
from __future__ import annotations
import gzip,json,shutil,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from crypto_beta_strategy_v2 import CFG,OUT,DATA,load_data,entry_labels,PHASES,csv
from src.crypto.beta_strategy import economic_regime,REGIMES,rebalance_targets
from src.crypto.market_beta import quantity_ledger

def main():
    # The first in-flight run predates compressed daily output; normalize once.
    plain=OUT/'contract_daily.csv';compressed=OUT/'contract_daily.csv.gz'
    if plain.exists():
        with plain.open('rb') as source,gzip.open(compressed,'wb') as target:shutil.copyfileobj(source,target)
        plain.unlink()
    hourly,prices,features,_=load_data()
    stored=pd.read_parquet(DATA/'signals.parquet');frozen=pd.read_parquet(DATA/'features.parquet')
    features={s:g.set_index('available_at').drop(columns='symbol') for s,g in frozen.groupby('symbol',sort=False)}
    sig={s:g.set_index('available_at').drop(columns='symbol') for s,g in stored.groupby('symbol',sort=False)}
    price=prices.loc[prices.index>=pd.Timestamp(CFG['first_forward'])+pd.Timedelta('5min')]
    index=price.index[:-1];rows=[];opportunities=[];transitions=[];signal_cards=[]
    moment_cards=[];factor_targets=[]
    for s,f in features.items():
        labels=entry_labels(f,prices[s]);state=economic_regime(f)
        changes=np.flatnonzero(np.r_[False,state[1:]!=state[:-1]])
        for phase,years in PHASES.items():
            phase_mask=(f.index>=pd.Timestamp(CFG['first_forward']))
            if years is not None:phase_mask&=f.index.year.isin(years)
            for st in range(6):
                m=phase_mask&(state==st)
                for h in [24,72,240]:
                    vals=labels.loc[m,f'r{h}'].dropna()
                    opportunities.append(dict(symbol=s,phase=phase,state=REGIMES[st],horizon=h,n=len(vals),mean_bp=float(vals.mean()*1e4),median_bp=float(vals.median()*1e4),positive_rate=float((vals>0).mean())))
            for col in ['market_beta','market_corr','variance_ratio72','risk_ratio','trend168']:
                vals=f.loc[phase_mask,col]
                moment_cards.append(dict(symbol=s,phase=phase,feature=col,mean=float(vals.mean()),median=float(vals.median()),q10=float(vals.quantile(.1)),q90=float(vals.quantile(.9))))
            for a in range(6):
                for b in range(6):
                    ix=changes[(state[changes-1]==a)&(state[changes]==b)&phase_mask[changes]]
                    if len(ix):
                        vals=labels.r24.iloc[ix].dropna()
                        transitions.append(dict(symbol=s,phase=phase,from_state=REGIMES[a],to_state=REGIMES[b],events=len(vals),return24_mean_bp=float(vals.mean()*1e4),positive_rate=float((vals>0).mean())))
        clock=f.index[(f.index>=pd.Timestamp(CFG['first_forward']))&(f.index+pd.Timedelta('5min')<price.index[-1])]
        local=f.reindex(clock);pos=index.get_indexer(clock+pd.Timedelta('5min'))
        strategies=dict(sig[s]);strategies['vol_long']=pd.Series(1.,index=f.index);strategies['hold']=pd.Series(1.,index=f.index)
        for policy,raw in strategies.items():
            signal=raw.reindex(clock).to_numpy();cap=np.minimum(1,CFG['vol_target']/(local.rv.to_numpy()*np.sqrt(8760)))
            desired=signal*cap if policy!='hold' else np.ones(len(clock))
            t,o=rebalance_targets(desired,clock.hour.to_numpy(),4,.12)
            if policy=='hold':t[:]=1;o[:]=False;o[0]=True
            target=np.zeros(len(index));order=np.zeros(len(index),dtype=bool);target[pos]=t;order[pos]=o
            v=quantity_ledger(price[[s]].to_numpy(),target,order,np.zeros((len(price),1)),np.ones(1),2)
            beginning=np.r_[1,v[:-1,5]]
            q=v[:,10]*beginning/price[s].to_numpy()[:-1]
            previous=np.r_[0,q[:-1]]
            positive_cost=2e-4*np.abs(q.clip(0)-previous.clip(0))*price[s].to_numpy()[:-1]
            negative_cost=2e-4*np.abs(np.minimum(q,0)-np.minimum(previous,0))*price[s].to_numpy()[:-1]
            positive_cost[-1]+=2e-4*max(0,q[-1])*price[s].iloc[-1]
            negative_cost[-1]+=2e-4*abs(min(0,q[-1]))*price[s].iloc[-1]
            long_money=v[:,8]*beginning-positive_cost;short_money=v[:,9]*beginning-negative_cost
            # Contribution fractions are exact dollars per initial account equity=1.
            for phase,years in PHASES.items():
                mask=np.ones(len(index),dtype=bool) if years is None else index.year.isin(years)
                first=np.flatnonzero(mask)[0];capital=beginning[first]
                long=float(long_money[mask].sum()/capital);short=float(short_money[mask].sum()/capital)
                total=float((v[mask][-1,5]-capital)/capital)
                long_exposure_hours=float(np.sum(np.maximum(q[mask],0)*price[s].to_numpy()[:-1][mask])/12/capital)
                short_exposure_hours=float(np.sum(np.abs(np.minimum(q[mask],0))*price[s].to_numpy()[:-1][mask])/12/capital)
                rows.append(dict(symbol=s,policy=policy,phase=phase,long_net_contribution=long,short_net_contribution=short,total_return=total,identity_error=abs(long+short-total),long_fee=float(positive_cost[mask].sum()/capital),short_fee=float(negative_cost[mask].sum()/capital),long_exposure_hours=long_exposure_hours,short_exposure_hours=short_exposure_hours,long_net_bp_per_exposure_hour=long/max(long_exposure_hours,1e-10)*1e4,short_net_bp_per_exposure_hour=short/max(short_exposure_hours,1e-10)*1e4))
            # Forecast-to-return association, separate from actual quantity drift.
            for phase,years in PHASES.items():
                mask=np.ones(len(clock),bool) if years is None else clock.year.isin(years)
                y=labels.r24.reindex(clock).to_numpy();good=mask&np.isfinite(y)
                a=signal[good];b=y[good]
                correlation=float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-9 else np.nan
                signal_cards.append(dict(symbol=s,policy=policy,phase=phase,signal_return24_ic=correlation,positive_prediction_fraction=float((a>0).mean()),short_prediction_fraction=float((a<0).mean()),direction_accuracy=float((np.sign(a)==np.sign(b)).mean()),mean_signal=float(a.mean()),std_signal=float(a.std())))
                market1=(entry_labels(local,prices.BTCUSDT).r1.to_numpy()+entry_labels(local,prices.ETHUSDT).r1.to_numpy())/2
                good1=mask&np.isfinite(market1)
                exposure=t[good1]*local.market_beta.to_numpy()[good1];next_return=market1[good1]
                covariance=float(np.mean((exposure-exposure.mean())*(next_return-next_return.mean())))
                factor_targets.append(dict(symbol=s,policy=policy,phase=phase,mean_target_market_beta=float(exposure.mean()),target_timing_covariance_bp_hour=covariance*1e4,average_target_drift_bp_hour=float(exposure.mean()*next_return.mean()*1e4)))
            print('Leg interpretation',s,policy,flush=True)
    legs=pd.DataFrame(rows)
    contract_daily=pd.read_csv(OUT/'contract_daily.csv.gz',parse_dates=['day'])
    portfolio_legs=[]
    for (policy,phase),g in legs.groupby(['policy','phase']):
        e=contract_daily[(contract_daily.policy==policy)&(contract_daily.fee_bp==2)].pivot(index='day',columns='symbol',values='equity')
        start_date=pd.Timestamp('2026-01-01' if phase=='reused_2026' else '2025-01-01' if phase=='year2025' else '2024-01-01',tz='UTC')
        past=e.loc[e.index<start_date]
        capital=past.iloc[-1] if len(past) else pd.Series(1.,index=e.columns)
        weights=capital/capital.sum();g=g.set_index('symbol').reindex(e.columns)
        row=dict(symbol='PORT12',policy=policy,phase=phase)
        for col in ['long_net_contribution','short_net_contribution','total_return','long_fee','short_fee','long_exposure_hours','short_exposure_hours']:
            row[col]=float((g[col]*weights).sum())
        row['identity_error']=abs(row['long_net_contribution']+row['short_net_contribution']-row['total_return'])
        row['long_net_bp_per_exposure_hour']=row['long_net_contribution']/max(row['long_exposure_hours'],1e-10)*1e4
        row['short_net_bp_per_exposure_hour']=row['short_net_contribution']/max(row['short_exposure_hours'],1e-10)*1e4
        portfolio_legs.append(row)
    legs=pd.concat([legs,pd.DataFrame(portfolio_legs)],ignore_index=True);csv(legs,'side_contribution.csv')
    csv(pd.DataFrame(opportunities),'state_future_opportunities.csv');csv(pd.DataFrame(transitions),'transition_future_opportunities.csv')
    csv(pd.DataFrame(signal_cards),'signal_transfer_cards.csv');csv(pd.DataFrame(moment_cards),'state_moment_cards.csv')
    csv(pd.DataFrame(factor_targets),'target_timing_covariance.csv')
    episodes=pd.read_csv(OUT/'episodes.csv')
    ep=[]
    for (symbol,policy,sign),g in episodes.groupby(['symbol','policy','sign']):
        complete=g.loc[~g.right_censored]
        ep.append(dict(symbol=symbol,policy=policy,sign=sign,n_completed=len(complete),n_censored=int(g.right_censored.sum()),mean_hours=float(complete.hours.mean()),median_hours=float(complete.hours.median()),q90_hours=float(complete.hours.quantile(.9)),fraction_12_264_hours=float(complete.hours.between(12,264).mean())))
    csv(pd.DataFrame(ep),'episode_cards.csv')
    summary=dict(all_12_contracts=sorted(s for s in legs.symbol.unique() if s!='PORT12'),max_leg_accounting_error=float(legs.identity_error.max()),primary_fee_bp=2,taker_fee_bp=5,forward_maturity=bool(json.loads((OUT/'run_manifest.json').read_text())['all_train_labels_matured']),status='strategy evidence, historical reused')
    (OUT/'diagnostic_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

if __name__=='__main__':main()
