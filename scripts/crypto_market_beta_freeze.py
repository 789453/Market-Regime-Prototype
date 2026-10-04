"""Final research artifact/source hashes and offline-report structural checks."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/crypto/market_beta_a_v1'

def main():
    report=(OUT/'MARKET_BETA_A_REPORT.html').read_text(encoding='utf-8')
    soup=BeautifulSoup(report,'html.parser')
    h=soup.find_all('h2');images=soup.find_all('img')
    assert len(h)==14 and len(images)==6
    ids={x.get('id') for x in h}
    internal=[a.get('href')[1:] for a in soup.select('nav a[href^="#"]')]
    assert len(internal)==14 and all(x in ids for x in internal)
    assert all(i.get('src','').startswith('data:image/png;base64,') for i in images)
    assert not soup.select('script[src],iframe,link[rel="stylesheet"]')
    scores=pd.read_csv(OUT/'policy_scores.csv')
    assert len(scores)==20*4*4 and scores.policy.nunique()==20
    verification=json.loads((OUT/'verification.json').read_text())
    assert verification['raw_hashes_unchanged'] and verification['all_label_maturity_pass']
    consistency=json.loads((OUT/'iteration_consistency.json').read_text())
    assert consistency['max_core_policy_return_change_after_passive_task_extension']<1e-10
    files=[ROOT/'configs/crypto_market_beta_a.json',ROOT/'src/crypto/market_beta.py',ROOT/'tests/test_crypto_market_beta.py',
        ROOT/'docs/MARKET_BETA_A_IMPLEMENTATION_2026_10_04.md',*sorted((ROOT/'scripts').glob('crypto_market_beta_*.py'))]
    files+=sorted(p for p in OUT.iterdir() if p.is_file() and p.name not in ['freeze_manifest.json','research_run.log'])
    text_suffixes={'.py','.md','.json','.html','.csv'}
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(
        p.read_bytes().replace(b'\r\n',b'\n') if p.suffix in text_suffixes else p.read_bytes()).hexdigest() for p in files}
    result=dict(frozen_on='2026-10-04',recorded_at=datetime.now(timezone.utc).isoformat(),
        branch='codex/market-beta-a',sha256=hashes,hash_mode='UTF-8 text canonical LF; binary bytes; raw source hashes in run_manifest are exact bytes',
        report_checks=dict(chapters=len(h),embedded_images=len(images),internal_navigation_valid=True,external_runtime_required=False,
            scientific_figures_visually_reviewed=6,html_source_structure_checked=True,browser_file_url_preview='blocked by browser protocol policy; local Codex file viewer used'),
        tests='82 passed',verification=verification,historical_status='development and reused exploration',
        independent_future_confirmation=False,real_execution_certified=False,live_orders_sent=False)
    (OUT/'freeze_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(hashed_files=len(hashes),chapters=len(h),images=len(images),policy_rows=len(scores),ledger_rows=verification['ledger_rows']),ensure_ascii=False))

if __name__=='__main__':
    main()
