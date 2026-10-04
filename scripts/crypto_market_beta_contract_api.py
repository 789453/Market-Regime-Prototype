"""Public contract metadata and historical settlement mark-price verification."""
import concurrent.futures as cf
import json
from pathlib import Path
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/crypto_market_beta_a.json').read_text())
OUT=ROOT/CFG['output_dir'];DATA=ROOT/CFG['data_dir']

def get_marks(symbol):
    rows=[]
    start=int(pd.Timestamp('2023-01-01',tz='UTC').timestamp()*1000)
    end=int(pd.Timestamp(CFG['historical_end_exclusive']).timestamp()*1000)-1
    while start<=end:
        response=requests.get('https://fapi.binance.com/fapi/v1/fundingRate',params=dict(symbol=symbol,startTime=start,endTime=end,limit=1000),timeout=20)
        response.raise_for_status()
        data=response.json()
        if not data:
            break
        rows.extend(data)
        later=max(int(r['fundingTime']) for r in data)+1
        if later<=start:
            raise ValueError('funding API made no progress')
        start=later
    frame=pd.DataFrame(rows)
    frame.to_parquet(DATA/f'funding_mark_{symbol}.parquet',index=False)
    return dict(symbol=symbol,status='ok',events=len(frame),nonempty_marks=int(pd.to_numeric(frame.markPrice,errors='coerce').notna().sum()))

def main():
    results=[]
    try:
        response=requests.get('https://fapi.binance.com/fapi/v1/exchangeInfo',timeout=20)
        response.raise_for_status()
        info=response.json()
        relevant=[r for r in info['symbols'] if r['symbol'] in CFG['market_symbols']]
        (OUT/'current_contract_metadata.json').write_text(json.dumps(dict(status='ok',symbols=relevant,source='public exchangeInfo; current metadata does not certify historical filters'),indent=2),encoding='utf-8')
    except Exception as exc:
        (OUT/'current_contract_metadata.json').write_text(json.dumps(dict(status='unavailable',reason=str(exc))),encoding='utf-8')
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(get_marks,s):s for s in CFG['market_symbols']}
        for f,s in futures.items():
            try:
                results.append(f.result())
            except Exception as exc:
                results.append(dict(symbol=s,status='unavailable',reason=str(exc)))
    (OUT/'settlement_mark_api_status.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    main()
