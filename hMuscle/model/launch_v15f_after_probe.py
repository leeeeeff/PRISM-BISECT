#!/usr/bin/env python3
"""
launch_v15f_after_probe.py
---------------------------
v_layer_probe_v15d_terms.py 완료 감지 → v15f_layer_select.py 레이어 매핑 자동 업데이트 → v15f 실행.

실행:
  cd hMuscle/model/
  python3 launch_v15f_after_probe.py &
"""

import os, sys, json, time, subprocess, re
from datetime import datetime

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROBE_JSON = os.path.join(BASE_DIR, '../../reports/layer_probe/layer_probe_v15d_terms_results.json')
V15F_PY    = os.path.join(BASE_DIR, 'v15f_layer_select.py')
LOG_DIR    = os.path.join(BASE_DIR, '../../logs_isoform')
POLL_SEC   = 30

# PLACEHOLDER terms that need probe results
PLACEHOLDER_TERMS = {
    'GO:0007204', 'GO:0045214', 'GO:0043161', 'GO:0042692',
    'GO:0055074', 'GO:0007517', 'GO:0032006', 'GO:0030048',
    'GO:0007268', 'GO:0007018', 'GO:0031175', 'GO:0030182',
}

def ts():
    return datetime.now().strftime('%H:%M:%S')

def extract_optimal_layers(probe_json_path):
    """probe JSON에서 GO term별 MLP peak layer 추출."""
    with open(probe_json_path) as f:
        data = json.load(f)

    mlp_auprc = data['mlp_auprc']  # {go_id: [auprc_L1, ..., auprc_L30]}
    result = {}
    print(f"\n[{ts()}] Probe results — MLP peak layers:")
    print(f"  {'GO term':<28} | MLP@peak | Peak layer | MLP@L30 | Gain")
    print(f"  {'-'*65}")
    for go_id, arr in mlp_auprc.items():
        valid = [(i, v) for i, v in enumerate(arr) if v is not None and v == v]
        if not valid:
            print(f"  {go_id}: no valid data — keeping placeholder")
            continue
        best_idx, best_val = max(valid, key=lambda x: x[1])
        opt_layer = best_idx + 1
        l30_val = arr[29] if arr[29] == arr[29] else 0.0
        gain = best_val - l30_val
        go_name = data['go_terms'].get(go_id, go_id)
        print(f"  {go_name:<28} | {best_val:.4f}   | L{opt_layer:02d}        | {l30_val:.4f}  | {gain:+.4f}")
        result[go_id] = opt_layer
    return result

def update_v15f_layer_mapping(new_layers: dict):
    """v15f_layer_select.py의 PLACEHOLDER 항목을 probe 결과로 교체."""
    with open(V15F_PY) as f:
        src = f.read()

    updated = src
    for go_id, opt_layer in new_layers.items():
        if go_id not in PLACEHOLDER_TERMS:
            continue
        # 패턴: '    'GO:XXXXXXX': NN,   # ... [PLACEHOLDER]'
        pattern = rf"('{re.escape(go_id)}':\s*)\d+(,\s*#.*?\[PLACEHOLDER\])"
        replacement = rf'\g<1>{opt_layer}\g<2>'
        new_src = re.sub(pattern, replacement, updated)
        if new_src == updated:
            print(f"  [WARN] Pattern not matched for {go_id} — manual check needed")
        else:
            updated = new_src

    # [PLACEHOLDER] → [PROBE] 표시 변경
    updated = updated.replace('[PLACEHOLDER]', '[PROBE]')

    with open(V15F_PY, 'w') as f:
        f.write(updated)
    print(f"  [{ts()}] v15f_layer_select.py updated.")

def launch_v15f():
    """GPU0에서 v15f 실행."""
    ts_str = datetime.now().strftime('%Y%m%d_%H%M')
    log_path = os.path.join(LOG_DIR, f'v15f_{ts_str}.log')
    cmd = (
        f'cd {BASE_DIR} && '
        f'CUDA_VISIBLE_DEVICES=0 '
        f'nohup conda run -n isoform_env python3 -u v15f_layer_select.py '
        f'> {log_path} 2>&1'
    )
    proc = subprocess.Popen(cmd, shell=True)
    print(f"\n[{ts()}] v15f launched — PID {proc.pid}")
    print(f"  Log: {log_path}")
    return proc, log_path

# ── Main polling loop ──────────────────────────────────────────────────────────
print(f"[{ts()}] Watching for probe completion...")
print(f"  Probe JSON: {PROBE_JSON}")
print(f"  Poll interval: {POLL_SEC}s")

prev_mtime = None
while True:
    if os.path.exists(PROBE_JSON):
        mtime = os.path.getmtime(PROBE_JSON)
        # 파일이 새로 생성되거나 업데이트됐을 때만 처리
        if prev_mtime != mtime:
            # JSON이 완전히 쓰여졌는지 확인 (파싱 시도)
            try:
                with open(PROBE_JSON) as f:
                    data = json.load(f)
                # 12개 term 모두 존재하고 L30 데이터까지 있는지 확인
                all_go = set(PLACEHOLDER_TERMS)
                present = set(data.get('mlp_auprc', {}).keys())
                if not all_go.issubset(present):
                    missing = all_go - present
                    print(f"[{ts()}] Probe JSON found but incomplete: missing {missing}")
                    prev_mtime = mtime
                    time.sleep(POLL_SEC)
                    continue
                # 각 term이 30개 레이어 데이터를 가지는지 확인
                mlp = data['mlp_auprc']
                complete = all(
                    len([v for v in mlp[g] if v is not None]) == 30
                    for g in all_go if g in mlp
                )
                if not complete:
                    print(f"[{ts()}] Probe JSON found but layers incomplete — waiting...")
                    prev_mtime = mtime
                    time.sleep(POLL_SEC)
                    continue
            except (json.JSONDecodeError, KeyError):
                print(f"[{ts()}] Probe JSON not yet parseable — waiting...")
                time.sleep(POLL_SEC)
                continue

            print(f"\n[{ts()}] === Probe complete! ===")
            new_layers = extract_optimal_layers(PROBE_JSON)
            print(f"\n[{ts()}] Updating v15f layer mapping ({len(new_layers)} terms)...")
            update_v15f_layer_mapping(new_layers)
            proc, log_path = launch_v15f()
            print(f"\n[{ts()}] v15f launched. Exiting watcher.")
            sys.exit(0)
        else:
            print(f"[{ts()}] Probe JSON unchanged — still waiting...")
    else:
        print(f"[{ts()}] Probe JSON not yet found — still waiting...")

    time.sleep(POLL_SEC)
