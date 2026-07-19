#!/usr/bin/env python3
"""
devils_c1_crosstab.py
C1 공격 (a): domain_diff<0(decoupled) subset이 canonical_is_lo 선택편향으로
오염됐는지 확인 — 만약 decoupled의 대부분이 canonical=짧은쪽(canonical_is_lo=1)
이면, domain_direction 자체가 편향된 부분집합을 보고 있는 것일 수 있다.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')

def load_domain_counts(tissue):
    path = 'domain_matrix_proper_test.npy' if tissue == 'muscle' else 'domain_matrix_brain_full.npy'
    dom = np.load(ROOT / 'hMuscle/results_isoform/features' / path)
    return dom.sum(axis=1).astype(np.int32)

def crosstab(tissue):
    df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv', sep='\t')
    dom_count = load_domain_counts(tissue)
    df = df[df['domain_binary'] == 1].copy()
    df['domain_diff'] = dom_count[df['long_idx'].to_numpy()] - dom_count[df['short_idx'].to_numpy()]

    pos = df[df['domain_diff'] > 0]
    neg = df[df['domain_diff'] < 0]

    print(f"\n=== {tissue} ===")
    print(f"aligned (domain_diff>0): n={len(pos)}")
    print(f"  canonical_is_lo=0 (canonical=긴쪽): {(pos['canonical_is_lo']==0).sum()}, "
          f"rate={100*(pos['canonical_is_lo']==0).sum()/len(pos):.1f}%")
    print(f"  canonical_is_lo=1 (canonical=짧은쪽): {(pos['canonical_is_lo']==1).sum()}, "
          f"rate={100*(pos['canonical_is_lo']==1).sum()/len(pos):.1f}%")

    print(f"\ndecoupled (domain_diff<0): n={len(neg)}")
    print(f"  canonical_is_lo=0 (canonical=긴쪽): {(neg['canonical_is_lo']==0).sum()}, "
          f"rate={100*(neg['canonical_is_lo']==0).sum()/len(neg):.1f}%")
    print(f"  canonical_is_lo=1 (canonical=짧은쪽): {(neg['canonical_is_lo']==1).sum()}, "
          f"rate={100*(neg['canonical_is_lo']==1).sum()/len(neg):.1f}%")

    # decoupled(domain_diff<0)에서 canonical=긴쪽(canonical_is_lo=0)만 따로 보면
    # severity_score>0 rate가 어떻게 변하나?
    neg_longcanon = neg[neg['canonical_is_lo'] == 0]
    if len(neg_longcanon) > 0:
        rate = np.mean(neg_longcanon['severity_score'] > 0)
        print(f"\n  decoupled 중 canonical=긴쪽만: n={len(neg_longcanon)}, severity_score>0 rate={rate:.3f}")

for tissue in ['muscle', 'brain']:
    crosstab(tissue)
