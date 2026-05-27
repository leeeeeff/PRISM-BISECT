# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def calculate_metrics(df, k_list=[1, 3]):
    """
    DataFrame(GeneID, IsoformID, Score, Label)을 받아 성능 지표 반환
    """
    y_true = df['Label'].values
    y_pred = df['Score'].values
    
    metrics = {}
    
    # 1. AUROC
    try:
        metrics['AUROC'] = roc_auc_score(y_true, y_pred)
    except:
        metrics['AUROC'] = 0.0
        
    # 2. AUPRC (불균형 데이터 핵심 지표)
    try:
        metrics['AUPRC'] = average_precision_score(y_true, y_pred)
    except:
        metrics['AUPRC'] = 0.0
        
    # 3. Gene-wise Top-k Accuracy
    grouped = df.groupby('GeneID')
    
    top_k_hits = {k: 0 for k in k_list}
    valid_genes_count = 0 
    
    for name, group in grouped:
        if sum(group['Label']) == 0:
            continue
            
        valid_genes_count += 1
        sorted_group = group.sort_values(by='Score', ascending=False)
        
        for k in k_list:
            top_k_samples = sorted_group.head(k)
            if sum(top_k_samples['Label']) > 0:
                top_k_hits[k] += 1
    
    if valid_genes_count > 0:
        for k in k_list:
            metrics['Gene_Top-%d_Acc' % k] = float(top_k_hits[k]) / valid_genes_count
    else:
        for k in k_list:
            metrics['Gene_Top-%d_Acc' % k] = 0.0
            
    metrics['Num_Valid_Genes'] = valid_genes_count

    return metrics


def compute_bias_score(df):
    """
    Gene-level bias score 측정 (score-variance 기반).

    GO term annotation이 gene-level이므로 label 기반 entropy는 degenerate.
    대신 예측 Score의 within-gene 분산으로 isoform-specificity를 측정:

        bias_score = mean(within_gene_score_std) / global_score_std

    - 0에 가까울수록: 같은 유전자의 isoform들이 동일한 score를 받음 → gene-level 예측
    - 1에 가까울수록: 같은 유전자 내에서도 score가 다름 → isoform-specific 예측
    - 기준: < 0.1 → gene-level shortcut 강함 [R2.1]
             > 0.3 → isoform-specific 특징 활용 중

    Args:
        df: DataFrame with columns [GeneID, IsoformID, Score, Label]
    Returns:
        dict with bias_score, within_gene_std, global_std, n_multi_gene
    """
    EPS = 1e-10

    global_std = df['Score'].std()

    # 유전자당 isoform이 2개 이상인 경우만 의미있음
    multi_iso_genes = df.groupby('GeneID').filter(lambda g: len(g) >= 2)
    n_multi = multi_iso_genes['GeneID'].nunique()

    if n_multi == 0:
        return {'bias_score': np.nan, 'within_gene_std': np.nan,
                'global_std': float(global_std), 'n_multi_gene': 0}

    within_stds = multi_iso_genes.groupby('GeneID')['Score'].std().dropna()
    mean_within_std = within_stds.mean()

    bias_score = mean_within_std / (global_std + EPS)

    # 추가: positive 유전자 내 score 분산 (이쪽이 더 중요)
    pos_genes = df[df['Label'] == 1]['GeneID'].unique()
    pos_df = multi_iso_genes[multi_iso_genes['GeneID'].isin(pos_genes)]
    pos_within_stds = pos_df.groupby('GeneID')['Score'].std().dropna()
    pos_bias = pos_within_stds.mean() / (global_std + EPS) if len(pos_within_stds) > 0 else np.nan

    return {
        'bias_score':      float(bias_score),   # 전체 기준
        'pos_bias_score':  float(pos_bias) if not np.isnan(pos_bias) else np.nan,  # positive 유전자만
        'within_gene_std': float(mean_within_std),
        'global_std':      float(global_std),
        'n_multi_gene':    int(n_multi),
    }
