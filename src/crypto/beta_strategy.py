"""Strategy mechanisms: causal filters, economic gates, and adaptive experts."""
from __future__ import annotations
import numpy as np
import pandas as pd
from numba import njit
from scipy.special import softmax

EXPERTS = ['fast', 'medium', 'slow', 'breakout', 'range', 'down_rebound', 'cash']
REGIMES = ['up_persistent','down_persistent','panic_continuation','rebound','range','transition']

def coin_features(hourly, market, breadth):
    p=np.log(hourly.close)
    r=p.diff()
    rv=np.sqrt(r.pow(2).ewm(halflife=168,min_periods=336).mean()).clip(lower=1e-6)
    f=pd.DataFrame(index=p.index)
    for h in [6,24,72,168,336]:
        f[f'trend{h}']=r.rolling(h).sum()/(rv*np.sqrt(h))
    for h in [24,72,168]:
        f[f'eff{h}']=r.rolling(h).sum()/r.abs().rolling(h).sum().replace(0,np.nan)
    f['risk_ratio']=np.sqrt(r.pow(2).rolling(24).mean())/rv
    f['down_share']=r.clip(upper=0).pow(2).rolling(24).sum()/r.pow(2).rolling(24).sum().replace(0,np.nan)
    f['dd168']=(p-p.rolling(168).max())/(rv*np.sqrt(168))
    f['recovery24']=(p-p.rolling(24).min())/(rv*np.sqrt(24))
    f['deviation72']=(p-p.ewm(span=72,adjust=False).mean())/(rv*np.sqrt(24))
    f['acceleration']=(r.rolling(6).sum()-r.shift(6).rolling(6).sum())/(rv*np.sqrt(6))
    f['volume_ratio']=hourly.quote_volume.rolling(24).mean()/hourly.quote_volume.rolling(168).mean()
    f['taker_balance']=hourly.taker_buy_quote_volume.rolling(24).sum()/hourly.quote_volume.rolling(24).sum()-.5
    f['market_trend72']=market.rolling(72).sum()/(np.sqrt(market.pow(2).ewm(halflife=168,min_periods=336).mean())*np.sqrt(72))
    f['breadth24']=breadth
    f['market_beta']=r.ewm(halflife=360,min_periods=720).cov(market)/market.ewm(halflife=360,min_periods=720).var().clip(lower=1e-10)
    f['market_corr']=r.rolling(720).corr(market)
    # Tradable Beta is a signed position. A risk state does not prescribe its sign.
    f['variance_ratio72']=r.rolling(72).sum().pow(2).rolling(720).mean()/(72*r.pow(2).rolling(720).mean())
    f['rv']=rv
    return f.replace([np.inf,-np.inf],np.nan)

def economic_regime(f):
    down=(f.trend168<-.6)&(f.trend24<-.35)
    rebound=(f.dd168<-.7)&(f.trend6>.35)&(f.recovery24>.65)&(f.acceleration>0)
    panic=down&(f.risk_ratio>1.25)&(f.trend6<0)&~rebound
    up=(f.trend168>.6)&(f.trend24>.2)&(f.eff72>.06)
    range_=(f.eff72.abs()<.07)&(f.trend168.abs()<.65)&(f.risk_ratio<1.3)
    return np.select([rebound,panic,down,up,range_],[3,2,1,0,4],default=5).astype(np.int64)

def expert_signals(f):
    e=pd.DataFrame(index=f.index)
    e['fast']=np.tanh(f.trend24/.8)
    e['medium']=np.tanh(f.trend72/.8)
    e['slow']=np.tanh((.6*f.trend168+.4*f.trend336)/.9)
    # Range-position incorporates completed current close; no unfinished high/low.
    e['breakout']=np.tanh((.7*f.trend72+.3*f.trend168)/.75)
    e['range']=-.7*np.tanh(f.deviation72)
    down=np.minimum(0,np.tanh((.65*f.trend24+.35*f.trend72)/.7))
    rebound=(f.dd168<-.7)&(f.trend6>.35)&(f.recovery24>.65)&(f.acceleration>0)
    down=np.where(rebound,.8*np.tanh(f.trend6),down)
    up=np.maximum(0,np.tanh((f.trend72+f.trend168)/1.6))
    e['down_rebound']=np.where(f.trend168<0,down,up)
    e['cash']=0.
    return e[EXPERTS].clip(-1,1)

def gate_weights(state):
    # prior strategy allocation: fast/medium/slow/breakout/range/down-rebound/cash
    priors=np.array([[.08,.24,.32,.22,.00,.12,.02],
                     [.12,.24,.25,.20,.00,.17,.02],
                     [.28,.15,.08,.15,.00,.32,.02],
                     [.30,.05,.02,.03,.20,.35,.05],
                     [.06,.05,.04,.05,.65,.05,.10],
                     [.12,.18,.28,.14,.08,.10,.10]])
    return priors[np.asarray(state)]

@njit(cache=True)
def rebalance_targets(desired,hours,interval=4,min_change=.12):
    target=np.zeros(len(desired));orders=np.zeros(len(desired),dtype=np.bool_)
    old=0.
    for i in range(len(desired)):
        if hours[i]%interval==0:
            new=desired[i]
            reversal=new*old<0 and abs(new)>.08
            exit_=abs(new)<.04 and abs(old)>=.04
            if abs(new-old)>=min_change or reversal or exit_:
                old=new;orders[i]=True
        target[i]=old
    return target,orders

@njit(cache=True)
def adaptive_experts(experts,state,prior,rewards,half_life_days=42.,temperature=4.):
    """reward[i] ended before i. Update only at i, then form position for i onward."""
    n,k=experts.shape
    sums=np.zeros((6,k));mass=np.zeros(6)
    weights=np.zeros((n,k));out=np.zeros(n)
    decay=np.exp(-np.log(2)/(half_life_days*24))
    for i in range(n):
        sums*=decay;mass*=decay
        if i>1:
            s=state[i-2]
            # Economic returns clipped per-hour: online performance is noisy.
            sums[s]+=rewards[i]
            mass[s]+=1.
        s=state[i]
        scores=temperature*24*sums[s]/(mass[s]+168.)
        scores-=scores.max()
        w=(.85*prior[i]+.15/k)*np.exp(scores)
        w/=w.sum()
        weights[i]=w
        out[i]=np.sum(w*experts[i])
    return out,weights

@njit(cache=True)
def markov_filter(emission,transition,initial):
    n,k=emission.shape
    out=np.zeros((n,k));p=initial.copy()
    for i in range(n):
        p=(p@transition)*emission[i]
        p=p/max(p.sum(),1e-300)
        out[i]=p
    return out

@njit(cache=True)
def kalman_drift(returns,rv,qratio=.003):
    """Local signed drift filtering; past RV sets the observation variance."""
    mu=0.;variance=1e-5;out=np.zeros(len(returns))
    for i in range(len(returns)):
        noise=max(rv[i]**2,1e-10)
        variance+=qratio*noise
        gain=variance/(variance+noise)
        mu+=gain*(returns[i]-mu)
        variance*=1-gain
        out[i]=np.tanh(mu/max(rv[i],1e-6)*np.sqrt(72))
    return out

def feature_registry():
    rows=[]
    definitions={
        'trend':'sum log return h / (EW hourly RMS sqrt(h))',
        'eff':'sum log return h / sum abs(log return) h',
        'risk_ratio':'24h RMS / 168h-half-life EW RMS',
        'down_share':'negative-return squared sum 24h / total squared sum',
        'dd168':'log close minus rolling maximum 168h / (EW RMS sqrt168)',
        'recovery24':'log close minus rolling minimum 24h / (EW RMS sqrt24)',
        'deviation72':'log close minus EMA72 / (EW RMS sqrt24)',
        'acceleration':'sum recent6h minus prior6h / (EW RMS sqrt6)',
        'volume_ratio':'mean quote volume24 / mean quote volume168',
        'taker_balance':'sum taker-buy quote24 / sum quote24 minus .5',
        'market_trend72':'BTC/ETH equal log returns trend72 normalized',
        'breadth24':'fraction of 12 completed 24h returns above zero',
        'market_beta':'EW cov(coin, BTC/ETH market)/EW var(market); half life360h min720h',
        'market_corr':'rolling correlation with BTC/ETH market720h',
        'variance_ratio72':'mean rolling72 sum-return squared720 / (72 mean r^2 720)',
        'rv':'EW sqrt(mean r^2), half-life168h min336h'}
    names=[f'trend{h}' for h in [6,24,72,168,336]]+[f'eff{h}' for h in [24,72,168]]+list(definitions)[2:]
    for name in names:
        family='trend' if name.startswith('trend') else 'eff' if name.startswith('eff') else name
        rows.append(dict(name=name,dimension=1,formula=definitions[family],unit='hourly fractional return' if name=='rv' else 'dimensionless',
                         window='see formula; trend/eff suffix is hours',direction='signed continuation for trend; risk/coherence not direction; others conditional',
                         duplicate='trend/eff/deviation correlated; coherence vs beta distinct',missing='NaN warmup, no forward-fill of unfinished bar',availability='completed hour t; executable t+5m'))
    return pd.DataFrame(rows)
