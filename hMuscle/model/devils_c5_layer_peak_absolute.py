#!/usr/bin/env python3
"""
devils_c5_layer_peak_absolute.py
C5 공격: rel_effect는 baseline≈0일 때 폭주 → 절대 delta로 peak 재확인.
만약 peak가 L12가 아니면 post-hoc cherry-pick.
"""
import pandas as pd
from pathlib import Path

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
df = pd.read_csv(ROOT / 'reports/severity_pairs/axis0_disorder_layer_profile.tsv', sep='\t')

for tissue in ['muscle', 'brain']:
    sub = df[df['tissue'] == tissue].copy()
    sub['abs_delta'] = sub['delta'].abs()
    peak_rel = sub.loc[sub['rel_effect'].abs().idxmax()]
    peak_abs = sub.loc[sub['abs_delta'].idxmax()]

    print(f"\n=== {tissue} ===")
    print(f"  Peak by |rel_effect|: L{int(peak_rel['layer'])}, rel={peak_rel['rel_effect']:+.2f}, delta={peak_rel['delta']:+.4f}")
    print(f"  Peak by |delta|:      L{int(peak_abs['layer'])}, delta={peak_abs['delta']:+.4f}, rel={peak_abs['rel_effect']:+.2f}")

    # L9-L17 범위에서 peak인지 확인
    mid_range = sub[(sub['layer'] >= 9) & (sub['layer'] <= 17)]
    peak_mid = mid_range.loc[mid_range['abs_delta'].idxmax()]
    print(f"  Peak within L9-L17:   L{int(peak_mid['layer'])}, delta={peak_mid['delta']:+.4f}")
