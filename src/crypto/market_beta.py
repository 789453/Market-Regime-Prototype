"""Market-level forecasts and physical-quantity linear-perpetual accounting.

All frame indexes denote availability, not bar-open times. No exchange orders.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from numba import njit

FEATURES = [
    ('trend24', 'sum(r,24)/(rv168*sqrt(24))', '24/168h', 'z', 'trend continuation', 'trend168, efficiency24'),
    ('trend168', 'sum(r,168)/(rv168*sqrt(168))', '168h', 'z', 'trend continuation', 'trend24, trend720'),
    ('trend720', 'sum(r,720)/(rv168*sqrt(720))', '720h', 'z', 'slow trend', 'trend168'),
    ('efficiency24', 'sum(r,24)/sum(abs(r),24)', '24h', 'ratio', 'signed persistence', 'trend24'),
    ('efficiency168', 'sum(r,168)/sum(abs(r),168)', '168h', 'ratio', 'signed persistence', 'trend168'),
    ('drawdown168', '(log_index-rolling_max(log_index,168))/(rv168*sqrt(168))', '168h', 'z', 'weakness or rebound context', 'rebound24'),
    ('rebound24', '(log_index-rolling_min(log_index,24))/(rv168*sqrt(24))', '24h', 'z', 'recovery; ambiguous', 'drawdown168'),
    ('risk_ratio', 'rv24/rv168', '24/168h', 'ratio', 'risk expansion; not direction', 'shock24'),
    ('down_share', 'sum(min(r,0)^2,24)/sum(r^2,24)', '24h', 'ratio', 'downside imbalance', 'trend24'),
    ('shock24', 'max(abs(r),24)/rv168', '24/168h', 'ratio', 'jump / reversal context', 'risk_ratio'),
    ('breadth24', 'mean_i[sum(r_i,24)>0]', '24h', 'fraction', 'common trend confirmation', 'trend24; selected 12-coin panel'),
    ('dispersion24', 'std_i(sum(r_i,24))/(rv168*sqrt(24))', '24h', 'ratio', 'fragmentation; ambiguous', 'breadth24'),
    ('correlation168', 'corr(r_BTC,r_ETH,168)', '168h', 'correlation', 'common-factor coherence', 'breadth24'),
    ('volume_ratio', 'mean(quote_volume,24)/mean(quote_volume,168)', '24/168h', 'ratio', 'activity confirmation', 'volume_divergence'),
    ('taker_balance', 'sum(taker_buy_quote,24)/sum(quote_volume,24)-0.5', '24h', 'ratio', 'buying pressure', 'trend24'),
    ('volume_divergence', 'efficiency24*(volume_ratio-1)', '24/168h', 'ratio', 'price/activity interaction', 'efficiency24, volume_ratio'),
    ('acceleration24', '(sum(r,6)-sum(r.shift(6),6))/(rv168*sqrt(6))', '12/168h', 'z', 'turning / acceleration', 'trend24'),
    ('trend_age', 'consecutive sign of sum(r,24), capped at 168 / 168', '24+168h', 'fraction', 'mature vs new trend; ambiguous', 'trend24'),
    ('range_position', '(log_index-min24)/(max24-min24)-0.5', '24h', 'ratio', 'breakout / failure context', 'rebound24'),
    ('known_funding', 'latest settled basket funding with 1h availability lag * 1e4', 'event, >=1h lag', 'bp/settlement', 'carry burden; not future realized rate', 'price trend'),
]

def feature_registry(lags=(0,3,6,12,24)):
    return pd.DataFrame([dict(name=f'{name}_lag{lag}', base=name, group=(
        'trend' if j<7 or j in [16,17,18] else 'risk' if j<10 else 'internal' if j<13 else 'activity' if j<16 else 'carry'),
        formula=formula, window=window, unit=unit, direction=direction, possibly_duplicate=duplicate,
        dimension=1, lag_hours=lag, missing='NaN => warmup/no forecast; zero-denominator safe except required rv; known funding 0 until first historical event',
        availability=f'completed hourly snapshot at t-{lag}h; funding settlement +1h; no current incomplete bar')
        for lag in lags for j,(name,formula,window,unit,direction,duplicate) in enumerate(FEATURES)])

def market_features(hourly, symbols, weights, context, known_funding=None):
    close = pd.concat({s:hourly[s]['close'] for s in context},axis=1)
    ret = np.log(close).diff()
    r = ret[symbols].mul(np.asarray(weights),axis=1).sum(axis=1,min_count=len(symbols))
    index = r.fillna(0).cumsum()
    rv = np.sqrt(r.pow(2).rolling(168,min_periods=168).mean()).clip(lower=1e-6)
    rv24 = np.sqrt(r.pow(2).rolling(24,min_periods=24).mean())
    quote = pd.concat({s:hourly[s]['quote_volume'] for s in symbols},axis=1).sum(axis=1,min_count=len(symbols))
    taker = pd.concat({s:hourly[s]['taker_buy_quote_volume'] for s in symbols},axis=1).sum(axis=1,min_count=len(symbols))
    f = pd.DataFrame(index=close.index)
    for h in (24,168,720):
        f[f'trend{h}'] = r.rolling(h,min_periods=h).sum()/(rv*np.sqrt(h))
    for h in (24,168):
        f[f'efficiency{h}'] = r.rolling(h).sum()/r.abs().rolling(h).sum().replace(0,np.nan)
    lo,hi = index.rolling(24).min(),index.rolling(24).max()
    f['drawdown168'] = (index-index.rolling(168).max())/(rv*np.sqrt(168))
    f['rebound24'] = (index-lo)/(rv*np.sqrt(24))
    f['risk_ratio'] = rv24/rv
    f['down_share'] = r.clip(upper=0).pow(2).rolling(24).sum()/r.pow(2).rolling(24).sum().replace(0,np.nan)
    f['shock24'] = r.abs().rolling(24).max()/rv
    r24 = ret.rolling(24,min_periods=24).sum()
    f['breadth24'] = (r24>0).mean(axis=1).where(r24.notna().all(axis=1))
    f['dispersion24'] = r24.std(axis=1)/(rv*np.sqrt(24))
    f['correlation168'] = ret[symbols[0]].rolling(168).corr(ret[symbols[1]])
    f['volume_ratio'] = quote.rolling(24).mean()/quote.rolling(168).mean().replace(0,np.nan)
    f['taker_balance'] = taker.rolling(24).sum()/quote.rolling(24).sum().replace(0,np.nan)-.5
    f['volume_divergence'] = f.efficiency24*(f.volume_ratio-1)
    f['acceleration24'] = (r.rolling(6).sum()-r.shift(6).rolling(6).sum())/(rv*np.sqrt(6))
    signs = np.sign(f.trend24.to_numpy())
    age = np.zeros(len(f))
    for i in range(1,len(age)):
        if np.isfinite(signs[i]) and signs[i]!=0:
            age[i] = min(168,age[i-1]+1) if signs[i]==signs[i-1] else 1
    f['trend_age'] = age/168
    f['range_position'] = ((index-lo)/(hi-lo).replace(0,np.nan)-.5).fillna(0)
    f['known_funding'] = 0 if known_funding is None else known_funding.reindex(f.index).ffill().fillna(0)*1e4
    f = f[[x[0] for x in FEATURES]].replace([np.inf,-np.inf],np.nan)
    return f, rv

def path_features(features, lags=(0,3,6,12,24)):
    # Hourly contract checked by the intake; shift means elapsed hours.
    return pd.concat([features.shift(l).rename(columns=lambda x:f'{x}_lag{l}') for l in lags],axis=1)

@njit(cache=True)
def path_labels(prices, entry, horizons, weights):
    n = len(entry)
    out = np.full((n,len(horizons)+3),np.nan)
    for i in range(n):
        e = entry[i]
        if e<0:
            continue
        for j in range(len(horizons)):
            end = e+int(horizons[j])*12
            if end<len(prices):
                out[i,j] = np.sum(weights*(prices[end]/prices[e]-1))
        end24=e+24*12
        if end24<len(prices):
            low,high,var = 0.,0.,0.
            for k in range(e+1,end24+1):
                value = np.sum(weights*(prices[k]/prices[e]-1))
                low,high=min(low,value),max(high,value)
                r = np.sum(weights*(prices[k]/prices[k-1]-1))
                var+=r*r
            out[i,len(horizons)],out[i,len(horizons)+1],out[i,len(horizons)+2]=-low,high,np.sqrt(var)
    return out

def soft_membership(x, centers, temperature):
    distance = ((x[:,None,:]-centers[None,:,:])**2).sum(axis=2)
    logits = -distance/max(temperature,1e-8)
    logits -= logits.max(axis=1,keepdims=True)
    q = np.exp(logits)
    return q/q.sum(axis=1,keepdims=True), distance

def decision_path(mu, risk_long, risk_short, uncertainty, rv, hours, cfg,
                  long_only=False, review_hours=None, risk_only=False, direct_score=None, initial_beta=0.):
    """Every hour a score; orders only on economic improvement or risk override.

    mu/risk/uncertainty are 24h-equivalent raw return units. The planner tracks
    its last target; the ledger records actual drifted holdings separately.
    """
    n = len(mu)
    target=np.zeros(n)
    trade=np.zeros(n,dtype=bool)
    reason=np.full(n,'hold',dtype=object)
    grid=np.asarray(cfg['action_grid'],dtype=float)
    if long_only:
        grid=grid[grid>=0]
    interval=review_hours or cfg['rebalance_hours']
    old=float(initial_beta)
    for t in range(n):
        cap=min(1.,cfg['annual_vol_target']/(max(rv[t],1e-6)*np.sqrt(8760)))
        actions=grid*cap
        stress=rv[t]*np.sqrt(8760)>1.5 or risk_long[t]>0.20 or risk_short[t]>0.20
        emergency=(stress and abs(old)>0.25) or abs(old)>cap+0.25
        if emergency:
            new=np.sign(old)*min(abs(old),0.25*cap)
            trade[t]=abs(new-old)>1e-9
            reason[t]='risk_override'
            old=new
        elif hours[t]%interval==0:
            if direct_score is not None:
                new=float(np.clip(direct_score[t],-cfg['max_short'],cfg['max_long'])*cap)
                if long_only:
                    new=max(0.,new)
                gain=(new-old)*mu[t]
            elif risk_only:
                new=cap
                gain=1.
            else:
                adverse=np.where(actions>=0,risk_long[t],risk_short[t])
                utility=actions*mu[t]-cfg['risk_penalty']*actions**2*adverse-cfg['uncertainty_penalty']*np.abs(actions)*uncertainty[t]
                oldrisk=risk_long[t] if old>=0 else risk_short[t]
                oldutility=old*mu[t]-cfg['risk_penalty']*old**2*oldrisk-cfg['uncertainty_penalty']*abs(old)*uncertainty[t]
                utility-=cfg['decision_fee_bp_side']/1e4*np.abs(actions-old)
                k=int(np.argmax(utility))
                new=actions[k]
                gain=utility[k]-oldutility
            change=abs(new-old)
            if direct_score is not None:
                accept=change>=cfg['minimum_action_change'] and (np.sign(new)==np.sign(old) or gain>cfg['switch_buffer_bp']/1e4+cfg['decision_fee_bp_side']/1e4*change)
            else:
                accept=change>=cfg['minimum_action_change'] and gain>cfg['switch_buffer_bp']/1e4
            if accept:
                trade[t]=True
                reason[t]='entry' if old==0 else 'exit' if new==0 else 'reverse' if old*new<0 else 'resize'
                old=new
        target[t]=old
    return target,trade,reason

@njit(cache=True)
def quantity_ledger(prices, targets, orders, funding, market_weights, fee_bp, liquidate=True):
    """Linear perp: equity changes only by q*dP, fees and signed funding.

    funding[t] settles at prices[t], using holdings brought into that timestamp
    BEFORE any new order. Funding mark is a disclosed trade-open proxy.
    Unchanged quantities do not imply any free rebalancing. Final row closes.
    """
    n=len(prices)-1
    coins=prices.shape[1]
    q=np.zeros(coins)
    equity=1.
    book=np.zeros((n,10+coins))
    fee=fee_bp/1e4
    for t in range(n):
        start=equity
        fund=np.sum(q*prices[t]*funding[t])
        equity-=fund
        cost=0.
        turn=0.
        if orders[t]:
            post=equity
            # Solve target fractions against equity after commission.
            for _ in range(12):
                desired=targets[t]*market_weights*post/prices[t]
                dollars=np.sum(np.abs(desired-q)*prices[t])
                post=equity-fee*dollars
            desired=targets[t]*market_weights*post/prices[t]
            dollars=np.sum(np.abs(desired-q)*prices[t])
            cost=fee*dollars
            turn=dollars/start
            q=desired
            equity-=cost
        w=q*prices[t]/start
        pnl=np.sum(q*(prices[t+1]-prices[t]))
        equity+=pnl
        if t==n-1 and liquidate:
            terminalfund=np.sum(q*prices[t+1]*funding[t+1])
            fund+=terminalfund
            equity-=terminalfund
            dollars=np.sum(np.abs(q)*prices[t+1])
            cost+=fee*dollars
            turn+=dollars/start
            equity-=fee*dollars
        if equity<=0 or start<=0:
            raise ValueError('equity exhausted: liquidation/margin not modeled')
        book[t,0]=equity/start-1
        book[t,1]=pnl/start
        book[t,2]=cost/start
        book[t,3]=fund/start
        book[t,4]=turn
        book[t,5]=equity
        book[t,6]=np.sum(w)
        book[t,7]=np.sum(np.abs(w))
        book[t,8]=np.sum(w[w>0]*(prices[t+1][w>0]/prices[t][w>0]-1))
        book[t,9]=np.sum(w[w<0]*(prices[t+1][w<0]/prices[t][w<0]-1))
        book[t,10:]=w
    return book

BOOK_COLUMNS=['net_return','price_return','fee_return','funding_return','turnover','equity','net_beta','gross','long_price','short_price']

def block_interval(values, block=14, repetitions=1000, seed=20261004):
    v=np.asarray(values,dtype=float)
    v=v[np.isfinite(v)]
    if not len(v):
        return np.nan,np.nan,np.nan
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,len(v),size=(repetitions,int(np.ceil(len(v)/block))))
    idx=(starts[:,:,None]+np.arange(block))%len(v)
    means=v[idx.reshape(repetitions,-1)[:,:len(v)]].mean(axis=1)
    return float(v.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))
