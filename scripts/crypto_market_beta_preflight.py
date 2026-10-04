"""Read-only provenance and official funding archive intake; no account access."""
from __future__ import annotations
import concurrent.futures as cf
import hashlib
import io
import json
import zipfile
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'configs/crypto_market_beta_a.json').read_text())
OUT = ROOT / CFG['output_dir']
CACHE = ROOT / CFG['data_dir'] / 'official_archives'

def archive(url):
    name = url.split('/data/')[-1].replace('/', '__')
    path = CACHE / name
    if not path.exists():
        response = requests.get(url, timeout=25)
        response.raise_for_status()
        path.write_bytes(response.content)
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    check = path.with_suffix(path.suffix + '.CHECKSUM')
    if not check.exists():
        response = requests.get(url + '.CHECKSUM', timeout=25)
        response.raise_for_status()
        check.write_text(response.text, encoding='utf-8')
    if digest != check.read_text().split()[0]:
        raise ValueError('official checksum mismatch: ' + url)
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        with z.open(z.namelist()[0]) as f:
            frame = pd.read_csv(f, header=None)
    return frame, digest

def check_price(symbol, month, product):
    prefix = 'spot' if product == 'spot' else 'futures/um'
    url = f'https://data.binance.vision/data/{prefix}/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip'
    result = dict(symbol=symbol, month=month, product=product, url=url)
    try:
        f, digest = archive(url)
        if not str(f.iloc[0,0]).isdigit():
            f = f.iloc[1:].reset_index(drop=True)
        f = f.apply(pd.to_numeric)
        stamp = f.iloc[:,0].astype('int64')
        if stamp.max() > 10**14:
            stamp //= 1000
        f.index = pd.to_datetime(stamp, unit='ms', utc=True)
        local = pd.read_parquet(Path(CFG['raw_root']) / symbol / '1h.parquet')
        local.index = pd.to_datetime(local.open_time, unit='ms', utc=True)
        common = local.index.intersection(f.index)
        errors = (local.loc[common,['open','high','low','close']].to_numpy() - f.loc[common,[1,2,3,4]].to_numpy()).max(axis=1)
        abs_error = abs(local.loc[common,['open','high','low','close']].to_numpy() - f.loc[common,[1,2,3,4]].to_numpy())
        result.update(status='ok', sha256=digest, rows=len(common), exact_ohlc_rows=int((abs_error.max(axis=1)<1e-8).sum()), max_abs_error=float(abs_error.max()))
    except Exception as e:
        result.update(status='unavailable', error=str(e))
    return result

def funding(symbol, month):
    url = f'https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip'
    result = dict(symbol=symbol, month=month, url=url)
    try:
        f, digest = archive(url)
        if not str(f.iloc[0,0]).isdigit():
            f.columns = f.iloc[0].astype(str)
            f = f.iloc[1:].copy()
        else:
            f.columns = ['calc_time','funding_interval_hours','last_funding_rate']
        f = f.apply(pd.to_numeric)
        f['settlement_at'] = pd.to_datetime(f.calc_time.astype('int64'), unit='ms', utc=True)
        f['symbol'] = symbol
        f = f[f.settlement_at < pd.Timestamp(CFG['historical_end_exclusive'])]
        f.to_parquet(ROOT / CFG['data_dir'] / f'funding_{symbol}_{month}.parquet', index=False)
        result.update(status='ok', sha256=digest, rows=len(f), first=str(f.settlement_at.min()), last=str(f.settlement_at.max()))
    except Exception as e:
        result.update(status='unavailable', error=str(e))
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    CACHE.mkdir(parents=True,exist_ok=True)
    # Two separated historical months, both products, every context symbol.
    tasks = [(s,m,p) for s in CFG['context_symbols'] for m in ['2023-01','2025-01'] for p in ['spot','usd_m']]
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        prices = list(pool.map(lambda a: check_price(*a), tasks))
    pd.DataFrame(prices).to_csv(OUT/'provenance_checks.csv', index=False, encoding='utf-8-sig')
    print('Price checks:', pd.DataFrame(prices).groupby(['product','status']).size().to_dict(),flush=True)
    months = pd.period_range('2023-01','2026-09',freq='M').astype(str)
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        funds = list(pool.map(lambda a: funding(*a), [(s,m) for s in CFG['market_symbols'] for m in months]))
    pd.DataFrame(funds).to_csv(OUT/'funding_archive_manifest.csv',index=False,encoding='utf-8-sig')
    frames = [pd.read_parquet(p) for p in (ROOT/CFG['data_dir']).glob('funding_*.parquet')]
    if frames:
        pd.concat(frames).sort_values(['settlement_at','symbol']).to_parquet(ROOT/CFG['data_dir']/'official_funding.parquet',index=False)
    good = [r for r in prices if r['status']=='ok' and r['product']=='usd_m' and r['rows']==r['exact_ohlc_rows']]
    evidence = {'matched_usdm_symbol_months':len(good),'required_symbol_months':24,
        'interpretation':'OHLC matches support USD-M trade-price provenance at audited months; not complete chain-of-custody certification',
        'account_fee_verified':False,'mark_price_verified':False,
        'funding_successful_archives':sum(r['status']=='ok' for r in funds),
        'funding_requested_archives':len(funds),'sources':['https://github.com/binance/binance-public-data','https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data','https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/account']}
    (OUT/'data_contract.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(evidence,ensure_ascii=False),flush=True)

if __name__ == '__main__':
    main()
