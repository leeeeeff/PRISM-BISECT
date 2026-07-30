#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_stage0_esmfold_pldct_audit.py
=================================
[Stage 0-lite] Structure-aware PLM decision gate (devils-advocate Vector 1).

Question: is the residual 30.2% non-domain within-gene "label-describability gap"
located in CONFIDENTLY-FOLDABLE regions (structure-aware PLM could help → NC path
viable) or in low-confidence / disordered regions (structure input unreliable →
structure-aware futile → GB)?

Method: reuse exp_brain_labelgap_rate.py pair machinery to build brain within-gene
2-isoform pairs, classify domain-affecting vs non-domain, then FOLD a sample with
ESMFold (local, no MSA) and measure per-residue pLDDT over the CHANGED (spliced) region.

Pre-registered decision gate (median pLDDT of NON-DOMAIN edited regions):
  > 70  → foldable → structure-aware pilot justified (Stage 1)
  < 60  → low-confidence → structure input unreliable there → GB direct
Positive control: DOMAIN-affecting edited regions should fold at HIGH pLDDT.

Constraint: cap measured-isoform length ≤ 500 aa for ESMFold tractability on shared
GPU (non-domain edits are short & N-terminal, so short proteins are well-represented;
noted as a sampling constraint, not a bias against the hypothesis).
"""
import os, re, glob, json, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict
import torch

RNG = np.random.default_rng(20260717)
DATA = Path('/home/welcome1/sw1686/DIFFUSE/hMuscle/data')
BRAIN = DATA / 'brain_isoquant_esm2/full'
FAA = Path('/home/welcome1/sw1686/DIFFUSE/reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')
DOMTBL = Path('/home/welcome1/sw1686/DIFFUSE/reports/truebrain_rerun_20260714/data/hmmscan_out')
OUT = Path('/home/welcome1/sw1686/DIFFUSE/reports/stage0_esmfold_audit')
OUT.mkdir(exist_ok=True)
_AA = set('ACDEFGHIKLMNPQRSTVWY')
IEVAL_MAX = 0.01
MAXLEN = 350       # lowered from 500: avoids long-seq CUDA OOM on shared GPU; both pools same regime
N_NONDOM = int(os.environ.get('N_NONDOM', 40))   # target non-domain pairs to fold
N_DOMCTL = int(os.environ.get('N_DOMCTL', 30))   # domain-affecting positive control

# ---- helpers copied from exp_brain_labelgap_rate.py (self-contained) --------
def clean(g): return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(' ', "")
def sani(s):
    s = s.replace('*', ''); return ''.join(c if c in _AA else 'X' for c in s)
def strip_orf(name): return re.sub(r'\.p\d+$', '', name)

def parse_faa():
    best = {}; cur_full, buf = None, []
    def flush():
        if cur_full is None: return
        seq = sani(''.join(buf)); base = strip_orf(cur_full)
        if base not in best or len(seq) > len(best[base][1]): best[base] = (cur_full, seq)
    for line in open(FAA):
        if line.startswith('>'):
            flush(); cur_full = line[1:].split()[0]; buf = []
        else: buf.append(line.strip())
    flush(); return best

def parse_domtbl():
    dom = defaultdict(list)
    for f in sorted(glob.glob(str(DOMTBL / '*.domtbl'))):
        for line in open(f):
            if line.startswith('#'): continue
            p = line.split()
            if len(p) < 23: continue
            try:
                ievalue = float(p[12]); q = p[3]; ef, et = int(p[19]), int(p[20])
            except (ValueError, IndexError): continue
            if ievalue <= IEVAL_MAX and et > ef: dom[q].append((ef, et))
    return dom

def overlaps(iv, doms):
    for (df, dt) in doms:
        if iv[0] < dt and df < iv[1]: return True
    return False

def changed_intervals(a, b):
    sm = SequenceMatcher(None, a, b, autojunk=False)
    iva, ivb, changed = [], [], 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal': continue
        changed += max(i2 - i1, j2 - j1)
        if i2 > i1: iva.append((i1, i2))
        if j2 > j1: ivb.append((j1, j2))
    return iva, ivb, changed

# ---- build pairs (identical classification to exp_brain_labelgap_rate.py) ----
def build_pairs():
    faa = parse_faa(); dom = parse_domtbl()
    print(f"faa bases {len(faa)}  domtbl ORFs {len(dom)}", flush=True)
    genes = np.array([clean(x) for x in np.load(BRAIN / 'brain_full_gene_names.npy', allow_pickle=True)])
    ids = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
    rec = {}
    for k, bid in enumerate(ids):
        base = strip_orf(clean(bid))
        if base in faa:
            full, seq = faa[base]
            if 20 <= len(seq) <= 5000: rec[k] = (full, seq, dom.get(full, []))
    gl, gi = np.unique(genes, return_inverse=True)
    cnt = np.bincount(gi, minlength=len(gl))
    nondom, domaff = [], []
    for g in np.where(cnt == 2)[0]:
        a, b = np.where(gi == g)[0]
        if a not in rec or b not in rec: continue
        (_, sa, da), (_, sb, db) = rec[a], rec[b]
        if sa == sb: continue
        iva, ivb, changed = changed_intervals(sa, sb)
        if changed == 0: continue
        hit = any(overlaps(iv, da) for iv in iva) or any(overlaps(iv, db) for iv in ivb)
        # measured isoform + its changed intervals (matches disorder computation)
        ivs, seq = (iva, sa) if iva else (ivb, sb)
        residues = sorted({i for (x, y) in ivs for i in range(x, y)})
        entry = {'gene': str(gl[g]), 'seq': seq, 'residues': residues,
                 'changed_aa': int(changed), 'len': len(seq)}
        (domaff if hit else nondom).append(entry)
    print(f"pairs: non-domain={len(nondom)}  domain-affecting={len(domaff)}", flush=True)
    return nondom, domaff

# ---- ESMFold (HuggingFace transformers, self-contained, no openfold) --------
def load_esmfold():
    from transformers import AutoTokenizer, EsmForProteinFolding
    tok = AutoTokenizer.from_pretrained('facebook/esmfold_v1')
    model = EsmForProteinFolding.from_pretrained('facebook/esmfold_v1',
                                                 low_cpu_mem_usage=True)
    model = model.eval().cuda()
    model.esm = model.esm.half()          # fp16 backbone → memory
    model.trunk.set_chunk_size(64)        # reduce activation memory for long seqs
    return model, tok

def fold_plddt(model, tok, seq):
    """Return per-residue pLDDT [0-100] (atom-masked mean) via transformers ESMFold."""
    ids = tok([seq], return_tensors='pt', add_special_tokens=False)['input_ids'].cuda()
    with torch.no_grad():
        out = model(ids)
    plddt = out['plddt'][0].float()               # [L, 37]
    mask = out['atom37_atom_exists'][0].float()   # [L, 37]
    per_res = (plddt * mask).sum(-1) / mask.sum(-1).clamp(min=1)
    per_res = per_res.cpu().numpy()
    if per_res.max() <= 1.0:                       # normalise if 0-1 scaled
        per_res = per_res * 100.0
    return per_res.astype(np.float32)

def audit(model, pool, label, n_target):
    elig = [e for e in pool if e['len'] <= MAXLEN and e['residues']]
    RNG.shuffle(elig)
    picked = elig[:n_target]
    print(f"\n[{label}] eligible(≤{MAXLEN}aa)={len(elig)}  folding {len(picked)}", flush=True)
    rows = []
    for i, e in enumerate(picked):
        t0 = time.time()
        try:
            pl = fold_plddt(model, TOK, e['seq'])
        except RuntimeError as ex:
            torch.cuda.empty_cache()
            print(f"  [{i+1}] {e['gene']} FOLD-FAIL {str(ex)[:60]}", flush=True); continue
        if len(pl) != e['len']:
            # length mismatch (rare) — align by min length
            m = min(len(pl), e['len'])
            res = [r for r in e['residues'] if r < m]
        else:
            res = e['residues']
        if not res: continue
        edit_plddt = float(np.mean([pl[r] for r in res]))
        rows.append({'gene': e['gene'], 'len': e['len'], 'changed_aa': e['changed_aa'],
                     'edit_plddt': round(edit_plddt, 2),
                     'whole_plddt': round(float(pl.mean()), 2)})
        print(f"  [{i+1:2d}/{len(picked)}] {e['gene']:14s} len={e['len']:4d} "
              f"edit_pLDDT={edit_plddt:5.1f} whole={pl.mean():5.1f} ({time.time()-t0:.0f}s)", flush=True)
        torch.cuda.empty_cache()
    return rows

def summarize(label, rows):
    ep = np.array([r['edit_plddt'] for r in rows])
    if len(ep) == 0:
        print(f"\n[{label}] no folds"); return {}
    s = {'n': len(ep), 'median_edit_plddt': float(np.median(ep)),
         'mean_edit_plddt': float(ep.mean()),
         'frac_gt70': float((ep > 70).mean()), 'frac_50_70': float(((ep >= 50) & (ep <= 70)).mean()),
         'frac_lt50': float((ep < 50).mean())}
    print(f"\n[{label}] n={s['n']}  median edit-pLDDT={s['median_edit_plddt']:.1f}  "
          f">70={s['frac_gt70']:.2f}  50-70={s['frac_50_70']:.2f}  <50={s['frac_lt50']:.2f}")
    return s

def main():
    print("=" * 70); print("  Stage 0-lite: ESMFold pLDDT audit of within-gene edited regions")
    print("=" * 70, flush=True)
    nondom, domaff = build_pairs()

    print("\nLoading ESMFold v1 via transformers (downloads ~5GB on first run)...", flush=True)
    global TOK
    model, TOK = load_esmfold()
    print("ESMFold loaded.", flush=True)

    rows_nd = audit(model, nondom, 'NON-DOMAIN', N_NONDOM)
    rows_dc = audit(model, domaff, 'DOMAIN-CTRL', N_DOMCTL)

    s_nd = summarize('NON-DOMAIN', rows_nd)
    s_dc = summarize('DOMAIN-CTRL', rows_dc)

    gate = ('STRUCTURE-PILOT JUSTIFIED (Stage 1)' if s_nd.get('median_edit_plddt', 0) > 70
            else 'STRUCTURE UNRELIABLE → GB' if s_nd.get('median_edit_plddt', 100) < 60
            else 'AMBIGUOUS (60-70) → judgement call')
    print("\n" + "=" * 70)
    print(f"  DECISION GATE (non-domain median edit-pLDDT = {s_nd.get('median_edit_plddt','NA')}): {gate}")
    print("=" * 70)

    json.dump({'maxlen': MAXLEN, 'nondomain': s_nd, 'domain_ctrl': s_dc, 'gate': gate,
               'rows_nondomain': rows_nd, 'rows_domain_ctrl': rows_dc},
              open(OUT / 'results.json', 'w'), indent=2)
    print(f"[Saved] {OUT/'results.json'}")

if __name__ == '__main__':
    main()
