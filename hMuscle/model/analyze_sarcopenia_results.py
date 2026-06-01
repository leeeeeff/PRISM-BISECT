"""
analyze_sarcopenia_results.py — 근감소증 GO term 평가 결과 자동 분석
실행: python analyze_sarcopenia_results.py
"""
import json, glob, sys
import numpy as np

# Find latest result file
files = sorted(glob.glob('../../reports/sarcopenia_eval/sarcopenia_final_*.json'))
if not files:
    files = sorted(glob.glob('../../reports/sarcopenia_eval/sarcopenia_partial_*.json'))
if not files:
    print("No result files found yet.")
    sys.exit(1)

with open(files[-1]) as f:
    data = json.load(f)

new_results  = data.get('new_results', [])
exist_results = data.get('existing_results', [])
all_results  = exist_results + new_results

print(f"Loaded: {files[-1]}")
print(f"New GO terms completed: {len(new_results)}/8")
print()

# ─── Summary table ─────────────────────────────────────────────────────────
COLS = ['go','name','type','n_pos_test','lr_auprc','v10b_mean','delta','sig','pos_bias']
print(f"{'GO':<15} {'Name':<32} {'T':>2} {'n':>5} {'LR':>8} {'v10-B':>8} {'Δ':>8} {'sig':>5} {'pb':>7}")
print("-"*78)

typeB_rows = []; typeA_rows = []
for r in all_results:
    v10b = r.get('v10b_mean', r.get('v10b_avg', float('nan')))
    delta = r.get('delta', float('nan'))
    pb = r.get('pos_bias') or float('nan')
    sig = r.get('sig','')
    t = r.get('type','?')
    mark = '★' if t == 'B' else ' '
    print(f"{r['go']:<15} {r.get('name','')[:31]:<32} {mark+t:>2} "
          f"{r.get('n_pos_test',0):>5} {r.get('lr_auprc',0):>8.4f} "
          f"{v10b:>8.4f} {delta:>+8.4f} {sig:>5} {pb:>7.3f}")
    if t == 'B': typeB_rows.append(r)
    else: typeA_rows.append(r)

print("-"*78)
if typeB_rows:
    b_v10b  = [r.get('v10b_mean', r.get('v10b_avg', 0)) for r in typeB_rows]
    b_lr    = [r.get('lr_auprc', 0) for r in typeB_rows]
    b_delta = [r.get('delta', 0) for r in typeB_rows]
    print(f"{'Type-B Macro':<50} {np.mean(b_lr):>8.4f} {np.mean(b_v10b):>8.4f} "
          f"{np.mean(b_delta):>+8.4f}")
if typeA_rows:
    a_v10b  = [r.get('v10b_mean', r.get('v10b_avg', 0)) for r in typeA_rows]
    a_lr    = [r.get('lr_auprc', 0) for r in typeA_rows]
    a_delta = [r.get('delta', 0) for r in typeA_rows]
    print(f"{'Type-A Macro':<50} {np.mean(a_lr):>8.4f} {np.mean(a_v10b):>8.4f} "
          f"{np.mean(a_delta):>+8.4f}")
print()

# ─── Key findings ──────────────────────────────────────────────────────────
print("KEY FINDINGS:")
sig_typeB = [r for r in typeB_rows if r.get('sig','') in ('*','**','***')]
print(f"  Type-B significant (p<0.05): {len(sig_typeB)}/{len(typeB_rows)}")
if sig_typeB:
    for r in sorted(sig_typeB, key=lambda x: -x.get('delta',0)):
        print(f"    {r['go']} ({r.get('name','')}): Δ={r['delta']:+.3f} {r['sig']}")

top_pb = sorted([r for r in all_results if r.get('pos_bias')],
                key=lambda x: -x.get('pos_bias', 0))[:5]
print(f"\n  Top pos_bias (genuine isoform discrimination):")
for r in top_pb:
    print(f"    {r['go']} ({r.get('name','')}): pos_bias={r['pos_bias']:.3f}")

# Sarcopenia highlight
sarc_go = {'GO:0006914','GO:0043161','GO:0032006'}
sarc_results = [r for r in new_results if r['go'] in sarc_go]
if sarc_results:
    print(f"\n  Sarcopenia core pathways:")
    for r in sarc_results:
        v10b = r.get('v10b_mean', r.get('v10b_avg', float('nan')))
        print(f"    {r['go']} ({r.get('name','')}): "
              f"v10-B={v10b:.4f} vs LR={r.get('lr_auprc',0):.4f} "
              f"Δ={r.get('delta',0):+.4f} {r.get('sig','')}")
