#!/usr/bin/env python3
"""
devils_c3_covariate_corr.py
C3 (b): domain_binary, size, resync_failure_binary가 axis3에 공동 집중하는 게
"3 covariate가 원래 상관"이기 때문인지 확인. 만약 셋의 상호상관이 낮은데
같은 축에 실리면 축이 진짜 다른 정보를 통합하는 것 — 높으면 conflation.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')

for tissue in ['muscle', 'brain']:
    df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv', sep='\t')
    dom = df['domain_binary'].to_numpy(float)
    size = np.log1p(df['size'].to_numpy(float))  # log-size
    resync = df['resync_failure_binary'].to_numpy(float)

    # Spearman correlation (size는 continuous, domain/resync는 binary)
    r_ds = stats.spearmanr(dom, size).correlation
    r_dr = stats.spearmanr(dom, resync).correlation
    r_sr = stats.spearmanr(size, resync).correlation

    print(f"\n=== {tissue} ===")
    print(f"  domain_binary ⇄ log(size):    ρ={r_ds:+.3f}")
    print(f"  domain_binary ⇄ resync_fail:  ρ={r_dr:+.3f}")
    print(f"  log(size) ⇄ resync_fail:      ρ={r_sr:+.3f}")
