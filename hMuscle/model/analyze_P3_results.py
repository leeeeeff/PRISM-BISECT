# -*- coding: utf-8 -*-
"""
analyze_P3_results.py

P3 / P3-256 실험 결과 즉시 분석 스크립트.
GPU 학습 완료 후 실행 → 자동으로 결과 수집·비교·Gate 판단.

사용법:
  python analyze_P3_results.py          # P3 + P3-256 모두 분석
  python analyze_P3_results.py --p3     # P3 only
  python analyze_P3_results.py --p3256  # P3-256 only
  python analyze_P3_results.py --live   # 진행 중 로그도 표시
"""

import os, sys, re, glob, argparse
import numpy as np
from datetime import datetime
from sklearn.metrics import average_precision_score, roc_auc_score

# ── 경로 설정 ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(SCRIPT_DIR, '../results_isoform')
LOGS_DIR     = os.path.join(SCRIPT_DIR, '../logs_isoform')

# ── GO terms ──────────────────────────────────────────────────────────────────
GO_TERMS = ['GO:0007204', 'GO:0030017', 'GO:0006941', 'GO:0003774', 'GO:0006096']

# ── 베이스라인 AUPRC ─────────────────────────────────────────────────────────
V8B_AUPRC = {
    'GO:0007204': 0.1462,
    'GO:0030017': 0.1570,
    'GO:0006941': 0.1177,
    'GO:0003774': 0.5686,
    'GO:0006096': 0.7945,
}
ESM2_LR_AUPRC = {
    'GO:0007204': 0.414,
    'GO:0030017': 0.561,
    'GO:0006941': 0.312,
    'GO:0003774': 0.825,
    'GO:0006096': 0.695,
}
# PCA dim sweep 결과 (esm2_dim_sweep.py에서 얻은 256-dim PCA LR 상한)
PCA256_AUPRC = {
    'GO:0007204': None,   # 실험 후 업데이트
    'GO:0030017': None,
    'GO:0006941': None,
    'GO:0003774': None,
    'GO:0006096': None,
}

# ── Experiment 태그 ───────────────────────────────────────────────────────────
EXPERIMENTS = {
    'P3':    'v8b-P3_integrated',
    'P3-256':'v8b-P3-256_integrated',
    'P3-512':'v8b-P3-512_integrated',  # SwissProt✗ + dim 512
    'D256':  'v8b-D256_integrated',    # SwissProt✓ + dim 256
}


# ─────────────────────────────────────────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────────────────────────────────────────

def safe_go(go_term):
    return go_term.replace(':', '_')


def find_latest_run_dir(go_term, ver_tag):
    """results_isoform/{GO}/{ver_tag}_*/ 중 가장 최신 디렉토리."""
    base = os.path.join(RESULTS_DIR, safe_go(go_term))
    pattern = os.path.join(base, '{}_{}_*'.format(ver_tag, safe_go(go_term)))
    # SAVE_DIR = {ver_tag}_{date_str} (date_str = YYYYMMDD_HHMM)
    pattern2 = os.path.join(base, '{}_*'.format(ver_tag))
    dirs = sorted(glob.glob(pattern2))
    if not dirs:
        return None
    return dirs[-1]   # 날짜 오름차순이므로 마지막이 최신


def parse_final_from_log(log_path):
    """[Final] AUROC=X AUPRC=X 파싱 → (auroc, auprc, meta_str) 또는 None."""
    if not os.path.exists(log_path):
        return None
    auroc, auprc, meta = None, None, ''
    with open(log_path) as f:
        for line in f:
            m = re.search(r'\[Final\] GO=(.+)', line)
            if m:
                meta = m.group(1).strip()
            m = re.search(r'\[Final\] AUROC=([0-9.]+)\s+AUPRC=([0-9.]+)', line)
            if m:
                auroc = float(m.group(1))
                auprc = float(m.group(2))
    if auprc is not None:
        return auroc, auprc, meta
    return None


def parse_inprogress_from_log(log_path):
    """[AUPRC] AUROC=X AUPRC=X (best=Y, patience=N/M) → 최신 줄 파싱."""
    if not os.path.exists(log_path):
        return None
    last = None
    with open(log_path) as f:
        for line in f:
            m = re.search(
                r'\[AUPRC\]\s+AUROC=([0-9.]+)\s+AUPRC=([0-9.]+)\s+\(best=([0-9.]+),\s*patience=(\d+)/(\d+)\)',
                line)
            if m:
                last = {
                    'auroc':   float(m.group(1)),
                    'auprc':   float(m.group(2)),
                    'best':    float(m.group(3)),
                    'patience':int(m.group(4)),
                    'max_pat': int(m.group(5)),
                }
    return last


def verify_auprc(run_dir, ver_tag, go_term):
    """Final_scores.txt + Final_labels.npy로 AUPRC 직접 재계산."""
    sg = safe_go(go_term)
    scores_path = os.path.join(run_dir, '{}_{}_Final_scores.txt'.format(ver_tag, sg))
    labels_path = os.path.join(run_dir, '{}_{}_Final_labels.npy'.format(ver_tag, sg))
    if not (os.path.exists(scores_path) and os.path.exists(labels_path)):
        return None, None
    labels = np.load(labels_path, allow_pickle=True)
    scores = []
    with open(scores_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    scores.append(float(parts[2]))
                except ValueError:
                    pass
    if len(scores) != len(labels):
        return None, None
    scores = np.array(scores)
    if labels.sum() == 0:
        return None, None
    ap   = average_precision_score(labels, scores)
    auroc= roc_auc_score(labels, scores)
    return ap, auroc


def find_runner_log(go_term, ver_prefix):
    """logs_isoform/{safe_go}/{ver_prefix}_*_Full.log 중 최신."""
    sg = safe_go(go_term)
    pattern = os.path.join(LOGS_DIR, sg, '{}_{}*_Full.log'.format(ver_prefix, sg))
    files = sorted(glob.glob(pattern))
    if files:
        return files[-1]
    # fallback: 날짜 없는 패턴
    pattern2 = os.path.join(LOGS_DIR, sg, '{}_*Full.log'.format(ver_prefix))
    files2 = sorted(glob.glob(pattern2))
    return files2[-1] if files2 else None


# ─────────────────────────────────────────────────────────────────────────────
# 실험 결과 수집
# ─────────────────────────────────────────────────────────────────────────────

def collect_results(exp_key, ver_tag, show_live=False):
    """
    Returns dict:
      go_term → {
        'status': 'done' | 'running' | 'missing',
        'auroc', 'auprc',                  ← [Final] 값
        'verified_auprc', 'verified_auroc', ← 재계산 값
        'meta', 'run_dir',
        'live': {...} or None              ← in-progress 정보
      }
    """
    results = {}
    for go in GO_TERMS:
        run_dir = find_latest_run_dir(go, ver_tag)
        entry = {'run_dir': run_dir, 'status': 'missing',
                 'auroc': None, 'auprc': None,
                 'verified_auprc': None, 'verified_auroc': None,
                 'meta': '', 'live': None}

        if run_dir:
            log_path = os.path.join(run_dir, '{}_{}_Full.log'.format(ver_tag, safe_go(go)))
            final = parse_final_from_log(log_path)
            if final:
                entry['auroc'], entry['auprc'], entry['meta'] = final
                entry['status'] = 'done'
                va, vauroc = verify_auprc(run_dir, ver_tag, go)
                entry['verified_auprc']  = va
                entry['verified_auroc']  = vauroc
            else:
                entry['status'] = 'running'
                live = parse_inprogress_from_log(log_path)
                if live:
                    entry['live'] = live

        # runner log fallback (in-progress)
        if entry['status'] != 'done' and show_live:
            ver_prefix = {'P3': 'v8b-P3', 'P3-256': 'v8b-P3-256', 'P3-512': 'v8b-P3-512'}.get(exp_key, ver_tag)
            rlog = find_runner_log(go, ver_prefix)
            if rlog and not entry['live']:
                live = parse_inprogress_from_log(rlog)
                if live:
                    entry['live'] = live

        results[go] = entry
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 출력 포매팅
# ─────────────────────────────────────────────────────────────────────────────

def print_experiment(exp_key, ver_tag, results):
    print("\n" + "=" * 80)
    print(" Experiment: {}  (ver_tag={})".format(exp_key, ver_tag))
    print("=" * 80)

    auprc_list = []

    print("{:<14} {:>8} {:>8} {:>8} {:>8} {:>8}  {}".format(
        "GO term", "AUPRC", "AUROC", "V.AUPRC", "v8b_ref", "LR_ref", "Status"))
    print("-" * 80)

    for go in GO_TERMS:
        e = results[go]
        v8b   = V8B_AUPRC.get(go, 0)
        lr    = ESM2_LR_AUPRC.get(go, 0)

        if e['status'] == 'done':
            auprc_str = "{:.4f}".format(e['auprc'])
            auroc_str = "{:.4f}".format(e['auroc'])
            vap = e['verified_auprc']
            vap_str = "{:.4f}".format(vap) if vap is not None else "  N/A"
            # delta vs v8b
            delta = e['auprc'] - v8b
            delta_str = "({:+.4f})".format(delta)
            status_str = "Done {}".format(delta_str)
            auprc_list.append(e['auprc'])
        elif e['status'] == 'running':
            auprc_str = "  ---"
            auroc_str = "  ---"
            vap_str   = "  ---"
            status_str = "Running"
            if e['live']:
                lv = e['live']
                status_str = "Running best={:.4f} pat={}/{}".format(
                    lv['best'], lv['patience'], lv['max_pat'])
        else:
            auprc_str = "  ---"
            auroc_str = "  ---"
            vap_str   = "  ---"
            status_str = "Missing"

        print("{:<14} {:>8} {:>8} {:>8} {:>8.4f} {:>8.4f}  {}".format(
            go, auprc_str, auroc_str, vap_str, v8b, lr, status_str))

    print("-" * 80)
    n_done = sum(1 for g in GO_TERMS if results[g]['status'] == 'done')
    if auprc_list:
        macro = np.mean(auprc_list)
        macro_str = "{:.4f} ({}/{} GO terms)".format(macro, n_done, len(GO_TERMS))
        v8b_macro = np.mean(list(V8B_AUPRC.values()))
        lr_macro  = np.mean(list(ESM2_LR_AUPRC.values()))
        print("{:<14} {:>8} {:>8} {:>8} {:>8.4f} {:>8.4f}  Macro({}/{})".format(
            "MACRO", "{:.4f}".format(macro), "", "",
            v8b_macro, lr_macro, n_done, len(GO_TERMS)))
        print()
        print("  v8b baseline  macro-AUPRC = {:.4f}".format(v8b_macro))
        print("  ESM-2 LR ref  macro-AUPRC = {:.4f}  ← Gate 기준".format(lr_macro))
        print("  {} macro-AUPRC = {:.4f}  ({:+.4f} vs v8b, {:+.4f} vs LR)".format(
            exp_key, macro, macro - v8b_macro, macro - lr_macro))
    else:
        print("  아직 완료된 GO term 없음 ({}/{})".format(n_done, len(GO_TERMS)))

    return auprc_list


def print_gate_decision(p3_auprc, p3256_auprc):
    """P3 Gate 판단 출력."""
    lr_macro  = np.mean(list(ESM2_LR_AUPRC.values()))  # 0.561
    v8b_macro = np.mean(list(V8B_AUPRC.values()))       # 0.357

    print("\n" + "=" * 80)
    print(" P3 Gate 판단")
    print("=" * 80)

    if p3_auprc:
        p3_macro = np.mean(p3_auprc)
        print("\n[P3] Macro-AUPRC = {:.4f}".format(p3_macro))
        if p3_macro > lr_macro:
            print("  → Case A: 파이프라인 유효 (> ESM-2 LR {:.3f})".format(lr_macro))
            print("     ✓ SwissProt 제거 단독으로 LR 상회 → 차원 확장으로 진행")
        elif p3_macro > v8b_macro + 0.05:
            print("  → Case B: 부분 개선 ({:.3f} ~ {:.3f})".format(v8b_macro, lr_macro))
            print("     → SwissProt 제거 효과 있음, 차원 확장 필수")
        elif p3_macro > v8b_macro:
            print("  → Case B-: 미미한 개선 ({:.4f} vs v8b {:.4f})".format(p3_macro, v8b_macro))
            print("     → SwissProt 효과 미미, 차원 문제가 지배적")
        else:
            print("  → Case C: 퇴화 ({:.4f} < v8b {:.4f})".format(p3_macro, v8b_macro))
            print("     ⚠ SwissProt 제거가 오히려 악영향 — devils-advocate 호출 필요")

    if p3256_auprc:
        p3256_macro = np.mean(p3256_auprc)
        print("\n[P3-256] Macro-AUPRC = {:.4f}".format(p3256_macro))
        if p3_auprc:
            p3_macro = np.mean(p3_auprc)
            dim_delta = p3256_macro - p3_macro
            print("  → P3 대비 dim 확장 기여: {:+.4f}".format(dim_delta))
            if p3256_macro > lr_macro:
                print("  → ✓ P3-256 LR 상회 → 차원 확장 유효")
            else:
                pct = p3256_macro / lr_macro * 100
                print("  → {:.1f}% of ESM-2 LR — 추가 개선 여지 있음".format(pct))
        if p3256_macro > lr_macro:
            print("\n  [GATE PASS] → 다음 단계: 5개 → 20개 GO term 확장, Bootstrap CI")
        elif p3256_macro > v8b_macro + 0.05:
            print("\n  [PARTIAL] → 차원 추가 확장 or Phase 2 개선 검토")
        else:
            print("\n  [REVIEW] → E2E fine-tuning 검토 또는 devils-advocate 호출")

    # swissProt 기여 vs dim 기여 분해
    if p3_auprc and p3256_auprc:
        p3m   = np.mean(p3_auprc)
        p3256m= np.mean(p3256_auprc)
        swiss_effect = p3m - v8b_macro
        dim_effect   = p3256m - p3m
        total_effect = p3256m - v8b_macro
        print("\n  [기여 분해]")
        print("    SwissProt 제거 효과: {:+.4f}  ({:.0f}%)".format(
            swiss_effect, swiss_effect/total_effect*100 if total_effect else 0))
        print("    Dim 64→256 효과:    {:+.4f}  ({:.0f}%)".format(
            dim_effect, dim_effect/total_effect*100 if total_effect else 0))
        print("    총 개선:             {:+.4f}".format(total_effect))


def print_per_go_detail(p3_res, p3256_res, d256_res=None, p3512_res=None):
    """GO term별 dim sweep 상세 비교표."""
    print("\n" + "=" * 104)
    print(" GO term별 dim sweep 비교 (AUPRC)  SP=SwissProt")
    print("=" * 104)
    print("{:<14} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}  Notes".format(
        "GO term", "v8b", "P3", "Δ(P3)", "P3-256", "P3-512", "D256", "LR"))
    print("{:<14} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "", "SP✓,64d", "SP✗,64d", "", "SP✗,256d", "SP✗,512d", "SP✓,256d", "640d"))
    print("-" * 104)

    selective_vals = []
    for go in GO_TERMS:
        v8b = V8B_AUPRC[go]
        lr  = ESM2_LR_AUPRC[go]

        p3r  = p3_res[go]   if p3_res   else {}
        p3ap = p3r.get('auprc')  if p3r.get('status')  == 'done' else None

        p256r  = p3256_res[go] if p3256_res else {}
        p256ap = p256r.get('auprc') if p256r.get('status') == 'done' else None

        p512r  = p3512_res[go] if p3512_res else {}
        p512ap = p512r.get('auprc') if p512r.get('status') == 'done' else None

        d256r  = d256_res[go] if d256_res else {}
        d256ap = d256r.get('auprc') if d256r.get('status') == 'done' else None

        p3_str   = "{:.4f}".format(p3ap)   if p3ap   is not None else "   ---"
        p256_str = "{:.4f}".format(p256ap) if p256ap is not None else "   ---"
        p512_str = "{:.4f}".format(p512ap) if p512ap is not None else "   ---"
        d256_str = "{:.4f}".format(d256ap) if d256ap is not None else "   ---"
        d3_str   = "{:+.4f}".format(p3ap - v8b) if p3ap is not None else "   ---"

        # selective best per GO term (max AUPRC across all experiments)
        avail = [v for v in [p3ap, p256ap, p512ap, d256ap] if v is not None]
        if avail:
            selective_vals.append(max(avail))

        notes = []
        if d256ap is not None and d256ap > lr:
            notes.append("D256>LR!")
        if p512ap is not None and p512ap > lr:
            notes.append("P512>LR!")
        if p256ap is not None and p256ap > lr:
            notes.append("P256>LR!")

        print("{:<14} {:>8.4f} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8.4f}  {}".format(
            go, v8b, p3_str, d3_str, p256_str, p512_str, d256_str, lr, " ".join(notes)))

    print("-" * 104)
    v8b_macro = np.mean(list(V8B_AUPRC.values()))
    lr_macro  = np.mean(list(ESM2_LR_AUPRC.values()))
    if d256_res:
        done = [d256_res[g].get('auprc') for g in GO_TERMS
                if d256_res[g].get('status') == 'done']
        if done:
            print("  D256  Macro ({}/5): {:.4f}  (v8b={:.4f}, LR={:.4f})".format(
                len(done), np.mean(done), v8b_macro, lr_macro))
    if p3512_res:
        done512 = [p3512_res[g].get('auprc') for g in GO_TERMS
                   if p3512_res[g].get('status') == 'done']
        if done512:
            print("  P3-512 Macro ({}/5): {:.4f}  (v8b={:.4f}, LR={:.4f})".format(
                len(done512), np.mean(done512), v8b_macro, lr_macro))
    if selective_vals and len(selective_vals) == len(GO_TERMS):
        sel_macro = np.mean(selective_vals)
        print("  Selective best Macro: {:.4f}  ({:.1f}% of LR)".format(
            sel_macro, sel_macro / lr_macro * 100))
    print("  LR ref: " + "  ".join("{:.4f}".format(ESM2_LR_AUPRC[g]) for g in GO_TERMS))


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p3',     action='store_true', help='P3 only')
    parser.add_argument('--p3256',  action='store_true', help='P3-256 only')
    parser.add_argument('--p3512',  action='store_true', help='P3-512 (SP✗+dim512) only')
    parser.add_argument('--d256',   action='store_true', help='D256 (SwissProt✓+dim256) only')
    parser.add_argument('--live',   action='store_true', help='Show in-progress logs')
    args = parser.parse_args()

    any_explicit = args.p3 or args.p3256 or args.p3512 or args.d256
    run_p3    = args.p3    or not any_explicit
    run_p3256 = args.p3256 or not any_explicit
    run_p3512 = args.p3512 or not any_explicit
    run_d256  = args.d256  or not any_explicit

    print("=" * 80)
    print(" P3 / P3-256 / P3-512 / D256 Result Analyzer  [{}]".format(
        datetime.now().strftime('%Y-%m-%d %H:%M')))
    print(" Results dir: {}".format(os.path.abspath(RESULTS_DIR)))
    print("=" * 80)

    p3_results   = None
    p3256_results= None
    p3512_results= None
    d256_results = None
    p3_auprc     = []
    p3256_auprc  = []
    p3512_auprc  = []
    d256_auprc   = []

    if run_p3:
        p3_results = collect_results('P3', EXPERIMENTS['P3'], show_live=args.live)
        p3_auprc   = print_experiment('P3', EXPERIMENTS['P3'], p3_results)

    if run_p3256:
        p3256_results = collect_results('P3-256', EXPERIMENTS['P3-256'], show_live=args.live)
        p3256_auprc   = print_experiment('P3-256', EXPERIMENTS['P3-256'], p3256_results)

    if run_p3512:
        p3512_results = collect_results('P3-512', EXPERIMENTS['P3-512'], show_live=args.live)
        p3512_auprc   = print_experiment('P3-512', EXPERIMENTS['P3-512'], p3512_results)

    if run_d256:
        d256_results = collect_results('D256', EXPERIMENTS['D256'], show_live=args.live)
        d256_auprc   = print_experiment('D256', EXPERIMENTS['D256'], d256_results)

    # 상세 비교 (dim sweep)
    if run_p3 or run_p3256 or run_p3512 or run_d256:
        print_per_go_detail(p3_results, p3256_results, d256_results, p3512_results)

    # Gate 판단
    if p3_auprc or p3256_auprc:
        print_gate_decision(p3_auprc if run_p3 else [],
                            p3256_auprc if run_p3256 else [])

    # 누락 정보 안내
    missing_p3   = [g for g in GO_TERMS if p3_results   and p3_results[g]['status']   != 'done'] if p3_results   else []
    missing_p256 = [g for g in GO_TERMS if p3256_results and p3256_results[g]['status'] != 'done'] if p3256_results else []
    missing_p512 = [g for g in GO_TERMS if p3512_results and p3512_results[g]['status'] != 'done'] if p3512_results else []
    missing_d256 = [g for g in GO_TERMS if d256_results  and d256_results[g]['status']  != 'done'] if d256_results  else []

    if missing_p3 or missing_p256 or missing_p512 or missing_d256:
        print("\n" + "-" * 80)
        if missing_p3:
            print("  P3 미완료:    {}".format(missing_p3))
        if missing_p256:
            print("  P3-256 미완료: {}".format(missing_p256))
        if missing_p512:
            print("  P3-512 미완료: {}".format(missing_p512))
        if missing_d256:
            print("  D256 미완료:  {}".format(missing_d256))
        print("  → 재실행하여 업데이트: python analyze_P3_results.py --live")

    print("\n")


if __name__ == '__main__':
    main()
