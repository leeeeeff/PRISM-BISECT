# -*- coding: utf-8 -*-
"""
v5-5 model logic pipeline diagram generator
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────────────────────────────────────────
C = {
    'data':      '#2C3E50',   # dark navy — data blocks
    'seq':       '#1A6B8A',   # teal-blue — sequence branch
    'domain':    '#1A5276',   # deep blue — domain branch
    'feature':   '#117A65',   # dark teal — feature model
    'loss':      '#884EA0',   # purple — loss functions
    'focal':     '#9B59B6',   # violet — focal loss
    'triplet':   '#7D3C98',   # dark violet — triplet loss
    'phase0':    '#7F8C8D',   # gray — phase 0
    'phase1':    '#E67E22',   # orange — phase 1
    'phase15':   '#F39C12',   # yellow-orange — phase 1.5
    'phase2':    '#C0392B',   # red — phase 2
    'phase3':    '#27AE60',   # green — phase 3
    'eval':      '#2980B9',   # blue — evaluation
    'embed':     '#16A085',   # green-teal — embedding
    'output':    '#2471A3',   # output blue
    'arrow':     '#5D6D7E',   # arrow gray
    'bg':        '#F8F9FA',   # background
    'header':    '#1C2833',   # header dark
    'anno':      '#922B21',   # annotation red
    'expr':      '#1E8449',   # expression green
}

fig = plt.figure(figsize=(28, 22), facecolor=C['bg'])
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 28)
ax.set_ylim(0, 22)
ax.axis('off')
ax.set_facecolor(C['bg'])


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def rounded_box(ax, x, y, w, h, color, alpha=0.92, lw=1.5, ls='-',
                ec=None, zorder=3, radius=0.18):
    ec = ec or color
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.0,rounding_size={radius}",
                         facecolor=color, edgecolor=ec,
                         linewidth=lw, linestyle=ls,
                         alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    return box


def text_center(ax, x, y, txt, size=8, color='white', weight='normal',
                zorder=5, ha='center', va='center', wrap=False):
    ax.text(x, y, txt, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=zorder,
            fontfamily='DejaVu Sans')


def arrow(ax, x1, y1, x2, y2, color=C['arrow'], lw=1.8, style='->', head=0.3, zorder=4):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=f'->', color=color,
                                lw=lw, mutation_scale=head*40),
                zorder=zorder)


def section_label(ax, x, y, txt, size=10, color=C['header']):
    ax.text(x, y, txt, fontsize=size, color=color, fontweight='bold',
            ha='left', va='center', zorder=6,
            fontfamily='DejaVu Sans')


def badge(ax, x, y, txt, bg='#E8DAEF', fc='#6C3483', size=6.5, zorder=6):
    ax.text(x, y, txt, fontsize=size, color=fc, fontweight='bold',
            ha='center', va='center', zorder=zorder,
            bbox=dict(boxstyle='round,pad=0.25', facecolor=bg,
                      edgecolor=fc, linewidth=1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────────────────────
rounded_box(ax, 0.3, 20.7, 27.4, 1.1, C['header'], alpha=0.95, radius=0.25)
text_center(ax, 14.0, 21.25,
            'v5-5 Integrated Full Model — Logic Pipeline',
            size=16, color='white', weight='bold')
text_center(ax, 14.0, 20.85,
            'Isoform Function Prediction  |  2-Modal (Seq + Domain)  |  4-Phase Training  |  Expression Label Propagation',
            size=9, color='#AED6F1')

# ─────────────────────────────────────────────────────────────────────────────
# ① DATA INPUTS  (y ≈ 18.3–20.3)
# ─────────────────────────────────────────────────────────────────────────────
section_label(ax, 0.4, 20.4, '① INPUT DATA', size=9, color=C['data'])

# Human / SwissProt Sequence
rounded_box(ax, 0.3, 18.4, 3.8, 1.7, C['seq'], alpha=0.88)
text_center(ax, 2.2, 19.6, 'Protein Sequences', size=9, weight='bold')
text_center(ax, 2.2, 19.2, 'Human (train) + SwissProt (aux)', size=7.5)
text_center(ax, 2.2, 18.82, 'amino-acid integer encoding', size=7)
text_center(ax, 2.2, 18.58, 'variable length → padded', size=7)

# Domain
rounded_box(ax, 4.4, 18.4, 3.0, 1.7, C['domain'], alpha=0.88)
text_center(ax, 5.9, 19.6, 'Protein Domains', size=9, weight='bold')
text_center(ax, 5.9, 19.2, 'Pfam domain integer IDs', size=7.5)
text_center(ax, 5.9, 18.82, 'dim = max_domain_count', size=7)
text_center(ax, 5.9, 18.58, 'mask_zero=True (LSTM)', size=7)

# GO Annotations
rounded_box(ax, 7.7, 18.4, 2.8, 1.7, C['anno'], alpha=0.82)
text_center(ax, 9.1, 19.6, 'GO Annotations', size=9, weight='bold')
text_center(ax, 9.1, 19.2, 'human_annotations.txt', size=7.5)
text_center(ax, 9.1, 18.82, 'swissprot_annotations.txt', size=7.5)
text_center(ax, 9.1, 18.58, '→ binary y_train, y_test', size=7)

# Expression (CPM)
rounded_box(ax, 10.8, 18.4, 3.0, 1.7, C['expr'], alpha=0.82)
text_center(ax, 12.3, 19.6, 'Expression (CPM)', size=9, weight='bold')
text_center(ax, 12.3, 19.2, 'CPM_transcript.txt', size=7.5)
text_center(ax, 12.3, 18.82, '24 samples → log1p', size=7.5)
text_center(ax, 12.3, 18.58, 'X_test_expr [N × 24]', size=7)

# Test isoform IDs
rounded_box(ax, 14.1, 18.4, 2.8, 1.7, C['data'], alpha=0.82)
text_center(ax, 15.5, 19.6, 'Test Isoform IDs', size=9, weight='bold')
text_center(ax, 15.5, 19.2, 'my_isoform_list_fixed.npy', size=7)
text_center(ax, 15.5, 18.82, 'my_gene_list_fixed.npy', size=7)
text_center(ax, 15.5, 18.58, 'my_sequence_matrix_fixed.npy', size=7)

badge(ax, 15.5, 18.3, '[READ-ONLY]', bg='#FDEDEC', fc='#C0392B', size=6)

# Upsample
rounded_box(ax, 17.2, 18.5, 2.5, 1.4, '#6E2F81', alpha=0.85)
text_center(ax, 18.45, 19.4, 'Upsample', size=9, weight='bold')
text_center(ax, 18.45, 19.05, 'isoform-level stratified', size=7.5)
text_center(ax, 18.45, 18.75, 'y_train_upsmp', size=7)

# ─────────────────────────────────────────────────────────────────────────────
# ② MODEL ARCHITECTURE  (y ≈ 14.5–18.0)
# ─────────────────────────────────────────────────────────────────────────────
section_label(ax, 0.4, 18.2, '② MODEL ARCHITECTURE', size=9, color=C['data'])

# --- Sequence branch ---
rounded_box(ax, 0.3, 15.6, 5.6, 2.4, '#1A5276', alpha=0.15, lw=1.5, ls='--',
            ec=C['seq'], radius=0.2)
text_center(ax, 3.1, 17.8, 'Sequence Branch', size=8.5, color=C['seq'], weight='bold')

seq_layers = [
    (0.5, 17.1, 2.0, 0.5, C['seq'], 'Embedding\n8001 → 32', 7.5),
    (0.5, 16.4, 2.0, 0.5, C['seq'], 'Conv1D(64, k=32)\n+ ReLU', 7.5),
    (0.5, 15.7, 2.0, 0.5, C['seq'], 'PyramidPooling\n[1, 2, 4, 8]', 7.5),
    (3.0, 17.1, 2.6, 0.5, C['seq'], 'Dense(32) + ReLU\n+ Dropout(0.2)', 7.5),
    (3.0, 16.4, 2.6, 0.5, C['seq'], 'Dense(16, relu)\nseq_feat [16]', 7.5),
]
for (bx, by, bw, bh, bc, bt, bs) in seq_layers:
    rounded_box(ax, bx, by, bw, bh, bc, alpha=0.85, radius=0.12)
    text_center(ax, bx + bw/2, by + bh/2, bt, size=bs)

# arrows within seq branch
for (x1, y1, x2, y2) in [
    (1.5, 17.1, 1.5, 16.9),
    (1.5, 16.4, 1.5, 16.2),
    (2.5, 17.35, 3.0, 17.35),
    (4.3, 17.1, 4.3, 16.9),
]:
    arrow(ax, x1, y1, x2, y2, color=C['seq'], lw=1.5, head=0.25)

# --- Domain branch ---
rounded_box(ax, 0.3, 13.6, 5.6, 1.8, '#1A3A5C', alpha=0.15, lw=1.5, ls='--',
            ec=C['domain'], radius=0.2)
text_center(ax, 3.1, 15.2, 'Domain Branch', size=8.5, color=C['domain'], weight='bold')

dom_layers = [
    (0.5, 13.8, 2.0, 0.5, C['domain'], 'Embedding\n→ 32, mask_zero', 7.5),
    (3.0, 13.8, 2.6, 0.5, C['domain'], 'LSTM(16)\ndomain_feat [16]', 7.5),
]
for (bx, by, bw, bh, bc, bt, bs) in dom_layers:
    rounded_box(ax, bx, by, bw, bh, bc, alpha=0.85, radius=0.12)
    text_center(ax, bx + bw/2, by + bh/2, bt, size=bs)
arrow(ax, 2.5, 14.05, 3.0, 14.05, color=C['domain'], lw=1.5)

# --- Concatenation + feature_model ---
rounded_box(ax, 6.2, 14.3, 3.0, 1.4, C['feature'], alpha=0.88, radius=0.18)
text_center(ax, 7.7, 15.25, 'Concatenate', size=9, weight='bold')
text_center(ax, 7.7, 14.95, '[seq_feat ‖ domain_feat]', size=7.5)
text_center(ax, 7.7, 14.65, '16 + 16 = 32-dim', size=7.5)
text_center(ax, 7.7, 14.42, 'feature_model output', size=7, color='#A9DFBF')
badge(ax, 7.7, 14.15, 'feature_model', bg='#D5F5E3', fc='#1E8449', size=6.5)

# seq → concat
arrow(ax, 4.3, 16.4, 4.3, 15.95)
arrow(ax, 4.3, 15.95, 6.2, 14.95, color=C['seq'], lw=1.5)
# domain → concat
arrow(ax, 4.3, 14.05, 6.2, 14.75, color=C['domain'], lw=1.5)

# --- Head (embedding + prediction) ---
rounded_box(ax, 9.5, 14.9, 2.5, 0.7, '#6C3483', alpha=0.88, radius=0.14)
text_center(ax, 10.75, 15.25, 'Dense(16) + ReLU', size=8)
text_center(ax, 10.75, 14.98, '+ Dropout(0.2)', size=7.5)

rounded_box(ax, 9.5, 14.0, 2.5, 0.65, C['embed'], alpha=0.9, radius=0.14)
text_center(ax, 10.75, 14.42, 'L2-Normalize', size=8, weight='bold')
text_center(ax, 10.75, 14.18, 'embedding_out [32]', size=7.5)

rounded_box(ax, 9.5, 13.0, 2.5, 0.7, '#E74C3C', alpha=0.88, radius=0.14)
text_center(ax, 10.75, 13.35, 'Dense(1, sigmoid)', size=8, weight='bold')
text_center(ax, 10.75, 13.08, 'prediction_out  [0, 1]', size=7.5)

# feature → head
arrow(ax, 9.2, 14.95, 9.5, 15.25, color=C['feature'], lw=1.5)
arrow(ax, 10.75, 14.9, 10.75, 14.65, color='#6C3483', lw=1.5)
arrow(ax, 10.75, 14.0, 10.75, 13.7, color=C['embed'], lw=1.5)

# base_model bracket
rounded_box(ax, 9.3, 12.75, 2.9, 3.25, '#884EA0', alpha=0.08, lw=1.5, ls='--',
            ec='#884EA0', radius=0.2)
text_center(ax, 10.75, 12.82, 'base_model', size=7, color='#884EA0')

# Phase 2 triplet model note
rounded_box(ax, 12.3, 14.0, 3.2, 1.1, '#7D3C98', alpha=0.82, radius=0.16)
text_center(ax, 13.9, 14.65, 'Triplet Model (Phase 2)', size=8, weight='bold')
text_center(ax, 13.9, 14.35, '3× base_model (A, P, N)', size=7.5)
text_center(ax, 13.9, 14.08, 'shared weights', size=7.5)
arrow(ax, 12.0, 14.35, 12.3, 14.35, color='#7D3C98', lw=1.5)

# ─────────────────────────────────────────────────────────────────────────────
# ③ TRAINING PHASES  (right side, y ≈ 5–13.5)
# ─────────────────────────────────────────────────────────────────────────────
section_label(ax, 0.4, 13.3, '③ TRAINING PHASES', size=9, color=C['data'])

# ── PHASE 0 ──────────────────────────────────────────────────────────────────
rounded_box(ax, 0.3, 11.5, 8.5, 1.55, C['phase0'], alpha=0.82, radius=0.2)
text_center(ax, 0.85, 12.85, 'Ph0', size=8, color='white', weight='bold')
text_center(ax, 3.5, 12.85, 'PHASE 0  —  Untrained Baseline', size=9, color='white', weight='bold')
text_center(ax, 4.55, 12.55, 'Save embeddings + prediction scores before any training', size=7.5, color='#D5DBDB')
text_center(ax, 4.55, 12.2, 'Metrics: AUROC, AUPRC, Silhouette, Centroid dist, Linear AUROC', size=7.5, color='#D5DBDB')
text_center(ax, 4.55, 11.87, 'Purpose: reference baseline for embedding quality evolution (no training)', size=7.5, color='#D5DBDB')

# ── PHASE 1 ──────────────────────────────────────────────────────────────────
rounded_box(ax, 0.3, 8.8, 13.5, 2.45, C['phase1'], alpha=0.82, radius=0.2)
text_center(ax, 0.85, 11.0, 'Ph1', size=8, color='white', weight='bold')
text_center(ax, 5.5, 11.0, 'PHASE 1  —  Embedding Triplet Training  (max 15 epochs)', size=9.5, color='white', weight='bold')

# sub-boxes inside phase 1
rounded_box(ax, 0.6, 9.0, 3.8, 1.65, '#F0B27A', alpha=0.25, lw=1.2, ls='--',
            ec='#F0B27A', radius=0.15)
text_center(ax, 2.5, 10.45, 'Triplet Loss', size=8, color='white', weight='bold')
text_center(ax, 2.5, 10.15, 'L = max(d(A,P) − d(A,N) + m, 0)', size=7.5, color='white')
text_center(ax, 2.5, 9.87, 'margin = 0.3 [I3]', size=7.5, color='#FAD7A0')
text_center(ax, 2.5, 9.6, 'distance = squared L2 on unit sphere', size=7, color='#FAD7A0')
text_center(ax, 2.5, 9.3, 'lr = 0.0005  |  GradientTape', size=7, color='#FAD7A0')

rounded_box(ax, 4.7, 9.0, 4.0, 1.65, '#F0B27A', alpha=0.25, lw=1.2, ls='--',
            ec='#F0B27A', radius=0.15)
text_center(ax, 6.7, 10.45, 'Semi-hard Negative Mining', size=8, color='white', weight='bold')
text_center(ax, 6.7, 10.15, 'Warmup (ep 1-2): random negatives', size=7.5, color='#FAD7A0')
text_center(ax, 6.7, 9.87, 'ep 3+: semi-hard  d(A,P) < d(A,N) < d(A,P)+m', size=7, color='white')
text_center(ax, 6.7, 9.6, 'fallback zone: d(A,N) < d(A,P) + 4m', size=7, color='#FAD7A0')
text_center(ax, 6.7, 9.3, 'refresh embeddings every 10 batches', size=7, color='#FAD7A0')

rounded_box(ax, 9.0, 9.0, 4.5, 1.65, '#F0B27A', alpha=0.25, lw=1.2, ls='--',
            ec='#F0B27A', radius=0.15)
text_center(ax, 11.25, 10.45, 'Dynamic n_batches [I4]', size=8, color='white', weight='bold')
text_center(ax, 11.25, 10.15, 'TARGET_COVERAGE = 6.0×/epoch', size=7.5, color='white')
text_center(ax, 11.25, 9.87, 'n = clip( ⌈n_pos × 6.0 / 256⌉, 20, 50 )', size=7, color='#FAD7A0')
text_center(ax, 11.25, 9.6, 'GO_0006936(840 pos) → n=20, cov=6.1×', size=7, color='#FAD7A0')
text_center(ax, 11.25, 9.3, 'GO_0006412(3046 pos) → n=50, cov=4.2×', size=7, color='#FAD7A0')

# Early stop conditions
text_center(ax, 6.5, 8.98, '[Early Stop]  active_rate < 2% for 4 consec. epochs  OR  margin_satisfied >= 60%',
            size=7, color='#FDEBD0')

# ── PHASE 1.5 ──────────────────────────────────────────────────────────────────
rounded_box(ax, 0.3, 6.7, 13.5, 1.85, C['phase15'], alpha=0.82, radius=0.2)
text_center(ax, 0.85, 8.35, 'Ph\n1.5', size=7.5, color='white', weight='bold')
text_center(ax, 6.5, 8.35, 'PHASE 1.5  —  Linear Probing  (Encoder Frozen,  2 epochs)', size=9.5, color='white', weight='bold')
text_center(ax, 6.5, 8.0, 'feature_model layers → frozen  |  Train only prediction head', size=7.5, color='#FEF9E7')
text_center(ax, 6.5, 7.7, 'Focal Loss: FL(p) = −0.25·(1−p)^2·log(p)  [γ=2, α=0.25]', size=7.5, color='white')
text_center(ax, 6.5, 7.4, 'lr = 0.001  |  batch_size ∈ {256, 512, 1024}  →  Unfreeze after 2 epochs', size=7.5, color='#FEF9E7')

badge(ax, 12.5, 8.35, 'Freeze → Unfreeze', bg='#FEF9E7', fc='#7D6608', size=7)

# ── PHASE 2 ──────────────────────────────────────────────────────────────────
rounded_box(ax, 0.3, 4.3, 13.5, 2.2, C['phase2'], alpha=0.82, radius=0.2)
text_center(ax, 0.85, 6.2, 'Ph2', size=8, color='white', weight='bold')
text_center(ax, 6.5, 6.2, 'PHASE 2  —  Joint Fine-tuning  (Focal + Triplet,  max 15 epochs)', size=9.5, color='white', weight='bold')

rounded_box(ax, 0.6, 4.5, 6.1, 1.45, '#F1948A', alpha=0.25, lw=1.2, ls='--',
            ec='#F1948A', radius=0.15)
text_center(ax, 3.65, 5.75, 'Combined Loss', size=8, color='white', weight='bold')
text_center(ax, 3.65, 5.45, 'L = 0.1 · L_triplet + 1.0 · L_focal', size=7.5, color='white')
text_center(ax, 3.65, 5.18, 'Focal: γ=2.0, α=0.10  (↓α vs Ph1.5)', size=7.5, color='#FADBD8')
text_center(ax, 3.65, 4.88, 'lr = 0.0003  |  batch_size = 64', size=7.5, color='#FADBD8')

rounded_box(ax, 7.0, 4.5, 6.5, 1.45, '#F1948A', alpha=0.25, lw=1.2, ls='--',
            ec='#F1948A', radius=0.15)
text_center(ax, 10.25, 5.75, 'AUPRC-based Early Stop [I2, I5, R9.1]', size=8, color='white', weight='bold')
text_center(ax, 10.25, 5.45, 'monitor: AUPRC on test set every 1 epoch', size=7.5, color='#FADBD8')
text_center(ax, 10.25, 5.18, 'checkpoint: save best AUPRC weights', size=7.5, color='white')
text_center(ax, 10.25, 4.88, 'patience = 3  →  restore best on stop', size=7.5, color='#FADBD8')

# ── PHASE 3 ──────────────────────────────────────────────────────────────────
rounded_box(ax, 0.3, 2.05, 13.5, 2.0, C['phase3'], alpha=0.82, radius=0.2)
text_center(ax, 0.85, 3.8, 'Ph3', size=8, color='white', weight='bold')
text_center(ax, 6.5, 3.8, 'PHASE 3  —  Test-time Expression Label Propagation  [I2]', size=9.5, color='white', weight='bold')

rounded_box(ax, 0.6, 2.22, 4.8, 1.55, '#A9DFBF', alpha=0.25, lw=1.2, ls='--',
            ec='#A9DFBF', radius=0.15)
text_center(ax, 3.0, 3.5, 'KNN Graph', size=8, color='white', weight='bold')
text_center(ax, 3.0, 3.22, 'X_test_expr [N × 24]  L2-normalized', size=7.5, color='white')
text_center(ax, 3.0, 2.95, 'k=15 neighbors, cosine similarity', size=7.5, color='#D5F5E3')
text_center(ax, 3.0, 2.68, 'threshold = 0.1  (zero weak edges)', size=7.5, color='#D5F5E3')

rounded_box(ax, 5.7, 2.22, 4.5, 1.55, '#A9DFBF', alpha=0.25, lw=1.2, ls='--',
            ec='#A9DFBF', radius=0.15)
text_center(ax, 7.95, 3.5, 'Score Propagation', size=8, color='white', weight='bold')
text_center(ax, 7.95, 3.22, 'prop = Σ sim(i,j) · score(j) / Σ sim', size=7.5, color='white')
text_center(ax, 7.95, 2.95, 'refined = (1−α)·base + α·prop', size=7.5, color='#D5F5E3')
text_center(ax, 7.95, 2.68, 'α ∈ {0.0, 0.2, 0.3, 0.5}', size=7.5, color='#D5F5E3')

rounded_box(ax, 10.5, 2.22, 3.0, 1.55, '#A9DFBF', alpha=0.25, lw=1.2, ls='--',
            ec='#A9DFBF', radius=0.15)
text_center(ax, 12.0, 3.5, 'Alpha Selection', size=8, color='white', weight='bold')
text_center(ax, 12.0, 3.22, 'AUPRC criterion [R9.1]', size=7.5, color='white')
text_center(ax, 12.0, 2.95, 'pick best_alpha by AUPRC', size=7.5, color='#D5F5E3')
text_center(ax, 12.0, 2.68, '→ final_scores output', size=7.5, color='#D5F5E3')

# Phase arrows (vertical)
for y_top, y_bot, color in [
    (11.5, 11.25, C['phase0']),
    (11.25, 8.8+2.45, C['phase1']),
    (8.8, 6.7+1.85, C['phase15']),
    (6.7, 4.3+2.2, C['phase2']),
    (4.3, 2.05+2.0, C['phase3']),
]:
    arrow(ax, 6.5, y_top, 6.5, y_bot, color=color, lw=2.0, head=0.25)

# ─────────────────────────────────────────────────────────────────────────────
# ④ EVALUATION & OUTPUT  (right column x ≈ 15–27.5)
# ─────────────────────────────────────────────────────────────────────────────
section_label(ax, 15.5, 13.3, '④ EVALUATION & OUTPUT', size=9, color=C['data'])

# Embedding quality box
rounded_box(ax, 15.5, 10.3, 11.8, 2.75, C['eval'], alpha=0.85, radius=0.2)
text_center(ax, 21.4, 12.78, 'Embedding Quality Analysis  [EMB]', size=9.5, weight='bold')

emb_metrics = [
    (15.8, 12.35, 'Silhouette Score (cosine)', 'cluster separation quality  > 0 = separable'),
    (15.8, 11.95, 'Intra-class Distance', 'mean sq-L2 among positive pairs  (↓ = tight)'),
    (15.8, 11.55, 'Inter-class Distance', 'mean sq-L2 pos vs neg pairs  (↑ = separated)'),
    (15.8, 11.15, 'Sep Ratio = inter / intra', 'embedding discriminability  (> 1 = good)'),
    (19.3, 12.35, 'Centroid Distance', '‖mean_pos − mean_neg‖  (L2)'),
    (19.3, 11.95, 'Linear AUROC', 'LogisticRegression on embeddings'),
    (19.3, 11.55, 'Pred AUROC / AUPRC', 'from model prediction scores  [R9.1]'),
    (19.3, 11.15, 'Phase coverage', 'Ph0 → Ph1 → Ph1.5 → Ph2 (4 phases)'),
]
for (mx, my, mname, mdesc) in emb_metrics:
    rounded_box(ax, mx, my - 0.17, 3.3, 0.37, '#1A5276', alpha=0.6, radius=0.08)
    text_center(ax, mx + 1.65, my + 0.01, mname, size=7.5, weight='bold')
    text_center(ax, mx + 1.65, my - 0.18, mdesc, size=6.8, color='#AED6F1')

text_center(ax, 21.4, 10.48, 'Computed at: Ph0 (untrained) · Ph1 (triplet) · Ph1.5 (linear) · Ph2 (joint)',
            size=7.5, color='#D6EAF8')

# UMAP visualization box
rounded_box(ax, 15.5, 7.9, 5.5, 2.15, '#2C3E50', alpha=0.85, radius=0.18)
text_center(ax, 18.25, 9.8, 'UMAP Visualization', size=9, weight='bold')
text_center(ax, 18.25, 9.48, '2×2 grid (4 phases × 2D UMAP)', size=7.5, color='#AED6F1')
text_center(ax, 18.25, 9.18, 'cosine metric, n_neighbors=15', size=7.5, color='#AED6F1')
text_center(ax, 18.25, 8.88, 'pos (red #e74c3c) vs neg (gray #cccccc)', size=7.5, color='#AED6F1')
text_center(ax, 18.25, 8.55, 'neg subsampled ≤5000 for speed', size=7.5, color='#AED6F1')
text_center(ax, 18.25, 8.15, '→ saved: {VER_TAG}_{GO}_umap.png', size=7, color='#7FB3D3')

# Score distribution box
rounded_box(ax, 21.3, 7.9, 6.0, 2.15, '#2C3E50', alpha=0.85, radius=0.18)
text_center(ax, 24.3, 9.8, 'Score Distribution', size=9, weight='bold')
text_center(ax, 24.3, 9.48, 'Histogram per phase (pos vs neg)', size=7.5, color='#AED6F1')
text_center(ax, 24.3, 9.18, 'AUROC + AUPRC annotated per panel', size=7.5, color='#AED6F1')
text_center(ax, 24.3, 8.88, 'Phase: Ph0, Ph1, Ph1.5, Ph2, Final(LP)', size=7.5, color='#AED6F1')
text_center(ax, 24.3, 8.55, 'density-normalized histogram', size=7.5, color='#AED6F1')
text_center(ax, 24.3, 8.15, '→ saved: {VER_TAG}_{GO}_score_dist.png', size=7, color='#7FB3D3')

# Final output
rounded_box(ax, 15.5, 5.5, 11.8, 2.15, '#186A3B', alpha=0.88, radius=0.2)
text_center(ax, 21.4, 7.38, 'FINAL OUTPUT', size=10, weight='bold')
out_items = [
    (15.7, 6.95, '{VER_TAG}_{GO}_Final_LabelProp_scores.txt',  'gene_id · iso_id · final_score (tsv)'),
    (15.7, 6.6,  '{VER_TAG}_{GO}_BaseModel_weights.h5',         'trained base_model weights'),
    (15.7, 6.25, '{VER_TAG}_{phase}_embeddings.npy',            '32-dim L2-normalized embeddings per phase'),
    (15.7, 5.9,  '{VER_TAG}_{phase}_scores.txt',                'raw prediction scores per phase'),
    (19.5, 6.95, 'Final_LabelProp_labels.npy',                  'y_test ground-truth labels'),
    (19.5, 6.6,  'training.log',                                 'full stdout log with all metrics'),
    (19.5, 6.25, 'Embedding quality summary table',              'all phases × all metrics printed'),
    (19.5, 5.9,  'AUPRC-best alpha logged',                      'alpha ∈ {0.0, 0.2, 0.3, 0.5}'),
]
for (ox, oy, oname, odesc) in out_items:
    rounded_box(ax, ox, oy - 0.14, 3.5, 0.34, '#1E8449', alpha=0.65, radius=0.08)
    text_center(ax, ox + 1.75, oy + 0.01, oname, size=6.8, weight='bold')
    text_center(ax, ox + 1.75, oy - 0.17, odesc, size=6.5, color='#A9DFBF')

# ─────────────────────────────────────────────────────────────────────────────
# KEY INNOVATIONS panel
# ─────────────────────────────────────────────────────────────────────────────
rounded_box(ax, 15.5, 2.05, 11.8, 3.2, '#1C2833', alpha=0.9, radius=0.22)
text_center(ax, 21.4, 5.0, 'KEY INNOVATIONS  (v5-5 vs earlier)', size=9.5, weight='bold', color='#F0F3F4')

innovations = [
    ('[I3] margin = 0.3',
     '"Frustrated triplet" effect: semi-hard zone stays wide → sustained gradients', '#F0B27A'),
    ('[I4] Dynamic n_batches',
     'clip(⌈n_pos × 6.0 / 256⌉, 20, 50) — per-GO-term coverage normalization', '#F8C471'),
    ('[I2] AUPRC criterion',
     'Phase 3 alpha & Phase 2 early-stop both optimize AUPRC, not AUROC  [R9.1]', '#82E0AA'),
    ('[I1] No prior bias init',
     'Removed: logit −4~−6 made focal gradient ≈ 0 for negatives', '#F1948A'),
    ('[I5] Phase 2 early stop',
     'patience=3 on AUPRC, check every epoch → prevent overfitting', '#AED6F1'),
]
for i, (tag, desc, col) in enumerate(innovations):
    y_i = 4.55 - i * 0.52
    rounded_box(ax, 15.75, y_i - 0.18, 11.3, 0.44, col, alpha=0.18,
                lw=1.0, ec=col, radius=0.1)
    text_center(ax, 17.55, y_i + 0.03, tag, size=8, color=col, weight='bold')
    text_center(ax, 21.65, y_i + 0.03, desc, size=7.5, color='#D5DBDB')

# ─────────────────────────────────────────────────────────────────────────────
# LEGEND
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    (C['seq'],     'Sequence branch (Conv1D + PyramidPool)'),
    (C['domain'],  'Domain branch (LSTM)'),
    (C['feature'], 'Feature model (2-modal concat)'),
    (C['embed'],   'Embedding (L2-normalized, 32-dim)'),
    (C['phase1'],  'Phase 1 — Triplet loss training'),
    (C['phase15'], 'Phase 1.5 — Linear probing (frozen)'),
    (C['phase2'],  'Phase 2 — Joint focal + triplet'),
    (C['phase3'],  'Phase 3 — Expression label prop'),
    (C['eval'],    'Evaluation & metrics'),
    (C['anno'],    'Protected / read-only data'),
]

# legend box
rounded_box(ax, 0.25, 0.15, 15.0, 1.65, '#1C2833', alpha=0.85, radius=0.2)
text_center(ax, 7.75, 1.6, 'LEGEND', size=8.5, weight='bold', color='#F0F3F4')

for i, (col, lbl) in enumerate(legend_items):
    col_i = i % 5
    row_i = i // 5
    lx = 0.55 + col_i * 3.0
    ly = 1.25 - row_i * 0.5
    rounded_box(ax, lx, ly - 0.12, 0.3, 0.28, col, alpha=0.9, radius=0.06)
    text_center(ax, lx + 1.35, ly + 0.02, lbl, size=6.8, color='#D5DBDB', ha='left')

# version watermark
text_center(ax, 27.5, 0.28, 'v5-5  |  2026-04-10', size=7, color='#7F8C8D', ha='right')

# ─────────────────────────────────────────────────────────────────────────────
# Connecting arrows: model arch → phases
# ─────────────────────────────────────────────────────────────────────────────
arrow(ax, 10.75, 13.0, 10.75, 12.6, color='#884EA0', lw=2.0)
arrow(ax, 10.75, 12.6, 6.5, 12.6, color='#884EA0', lw=2.0)
arrow(ax, 6.5, 12.6, 6.5, 11.5, color='#884EA0', lw=2.0)

# phase3 → output
arrow(ax, 13.8, 3.05, 15.5, 6.5, color=C['phase3'], lw=1.8)

# eval arrow from phase2
arrow(ax, 13.8, 5.4, 15.5, 10.5, color=C['phase2'], lw=1.5)

plt.savefig('/home/welcome1/sw1686/DIFFUSE/hMuscle/model/v5-5_pipeline_diagram.png',
            dpi=150, bbox_inches='tight', facecolor=C['bg'])
print("Saved: v5-5_pipeline_diagram.png")
