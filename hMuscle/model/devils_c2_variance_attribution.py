#!/usr/bin/env python3
"""
devils_c2_variance_attribution.py
C2 CRITICAL 공격: axis0(evr=.156)은 분산 최대 축. occlusion이 covariate 상관을
더 크게 흔드는 건 "causal usage"가 아니라 "제거된 분산이 크기 때문"일 수 있다.
random direction은 저분산인데 axis0은 고분산 → 불공정 비교.

CHECK: axis0 방향이 L15/L30 zscore된 임베딩에서 설명하는 분산(evr_within_layer)
vs random direction(동일 640-dim space)이 설명하는 분산. 만약 axis0이 random보다
월등히 크면 delta_real > random-null이 "causal usage"가 아니라 "분산 크기 confound".
"""
import numpy as np
from pathlib import Path

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
N_RANDOM = 20
SEED = 42

def load_layer_stats():
    W = np.load(ROOT / 'reports/v20b_pca_interp/W_axes_8x640.npy')
    mu = np.load(ROOT / 'reports/v20b_pca_interp/layer_stats_mu.npy')
    sd = np.load(ROOT / 'reports/v20b_pca_interp/layer_stats_sd.npy')
    return W, mu, sd

def load_raw_layer(tissue, layer):
    if tissue == 'muscle':
        return np.load(ROOT / f'hMuscle/data/esm2_layer_{layer:02d}_t30_150M.npy').astype(np.float32)
    else:
        return np.load(ROOT / f'hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_layer{layer:02d}_t30_150M.npy').astype(np.float32)

def evr_of_direction(z, direction):
    """분산 설명 비율 = var(z @ direction) / total_var(z)."""
    proj = z @ direction
    var_proj = np.var(proj)
    var_total = np.var(z)  # per-feature variance의 평균(trace)
    return var_proj / var_total if var_total > 0 else 0.0

W, mu, sd = load_layer_stats()
w0 = W[0] / np.linalg.norm(W[0])

rng = np.random.default_rng(SEED)
random_dirs = rng.normal(size=(N_RANDOM, 640))
random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)

for tissue in ['muscle', 'brain']:
    print(f"\n{'='*60}\n{tissue}\n{'='*60}")
    for layer in [15, 30]:
        raw = load_raw_layer(tissue, layer)
        z = (raw - mu[layer-1]) / sd[layer-1]  # (N_iso, 640)

        evr_axis0 = evr_of_direction(z, w0)
        evr_random = [evr_of_direction(z, d) for d in random_dirs]
        evr_r_mean = np.mean(evr_random)
        evr_r_std = np.std(evr_random)

        print(f"\nLayer {layer}:")
        print(f"  axis0 분산 설명 비율(within-layer z-scored emb): {evr_axis0:.6f}")
        print(f"  20 random directions: mean={evr_r_mean:.6f}, std={evr_r_std:.6f}")
        print(f"  axis0 / random_mean ratio = {evr_axis0/evr_r_mean:.2f}")
        # joint-PCA evr은 전체 30층×640=19200차원 공간 기준이었는데, 여기선 single-layer
        # 640차원만 보므로 evr 수치가 다를 수 있다 — 하지만 axis0 vs random의 "상대비"는
        # occlusion null의 공정성을 검증하는 데 충분.
