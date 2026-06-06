"""Rule-based interpretation engine — structured report dicts.

Each interpret_* returns:
{
  'headline': str,            # 1문장 핵심 발견 (수치 포함)
  'bullets': list[str],       # 3-6개 수치 기반 관찰 (구체적)
  'interpretation': str,      # 2-4문장 생물학적 맥락
  'caveats': list[str],       # 해석 주의사항
  'next_steps': list[str],    # 권장 다음 단계 (구체적)
  'markdown': str,            # 다운로드용 마크다운
}
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional


# ── Tissue context ────────────────────────────────────────────────────────────

_TISSUE_CONTEXT = {
    'muscle': {
        'name': '골격근',
        'key_functions': ['근육 수축(GO:0006936)', '에너지 대사(GO:0006096)', '미토콘드리아 전자전달(GO:0022900)'],
        'reference_auprc': 0.7022,
        'known_genes': ['MYH7', 'TNNT3', 'NDUFS4', 'ATP2A1'],
    },
    'brain': {
        'name': '뇌 (18-GO 패널, zero-shot)',
        'key_functions': ['시냅스 전달(GO:0007268)', '신경발생(GO:0022008)', '산화적 인산화(GO:0006119)'],
        'reference_auprc': 0.5998,
        'known_genes': ['KIF21B', 'NDUFS4', 'DLG1', 'PTPRF'],
    },
    'brain_extended': {
        'name': '뇌 (73-GO 확장 패널)',
        'key_functions': ['시냅스 가소성', '미엘린화', 'GABA 신호'],
        'reference_auprc': 0.58,
        'known_genes': ['KIF21B', 'NDUFS4', 'APP', 'CLU'],
    },
    'brain_672': {
        'name': '뇌 전체 (672-GO 패널, 44 모듈)',
        'key_functions': ['미토콘드리아 호흡', '시냅스 전달', '세포 이동'],
        'reference_auprc': None,
        'known_genes': ['NDUFS4', 'NDUFS7', 'NDUFS8', 'KIF21B', 'DLG1', 'PTPRF'],
    },
    'muscle_only': {
        'name': '골격근 학습 전용',
        'key_functions': ['근육 수축', '에너지 대사'],
        'reference_auprc': 0.7022,
        'known_genes': ['MYH7', 'TNNT3'],
    },
}


# ── Shared helpers ────────────────────────────────────────────────────────────

def _pct(n: int, total: int, denom_fallback: int = 1) -> str:
    d = max(1, total) if total else denom_fallback
    return f"{100 * n / d:.1f}%"


def _top_names(series: pd.Series, n: int = 3) -> list[str]:
    return series.dropna().astype(str).tolist()[:n]


def _to_markdown(section: str, headline: str, bullets: list[str],
                 interpretation: str, caveats: list[str],
                 next_steps: list[str]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    bmd = "\n".join(f"- {b}" for b in bullets)
    cmd = "\n".join(f"- ⚠️ {c}" for c in caveats)
    smd = "\n".join(f"- [ ] {s}" for s in next_steps)
    return (
        f"## PRISM 분석 리포트 — {section}\n\n"
        f"*생성: {ts}*\n\n"
        f"**핵심 발견**: {headline}\n\n"
        f"**주요 관찰**\n{bmd}\n\n"
        f"**생물학적 해석**\n{interpretation}\n\n"
        f"**해석 주의사항**\n{cmd}\n\n"
        f"**권장 다음 단계**\n{smd}\n"
    )


# ── QC & Overview ─────────────────────────────────────────────────────────────

def interpret_qc(
    cfg: dict,
    classified_df: Optional[pd.DataFrame] = None,
    coverage_rep=None,
    novel_rep=None,
    val_rep=None,
) -> dict:
    sm      = cfg.get('score_matrix')
    thr     = cfg.get('score_threshold', 0.4)
    tissue  = cfg.get('tissue', 'unknown')
    tctx    = _TISSUE_CONTEXT.get(tissue, {})
    n_iso   = sm.shape[0] if sm is not None else 0
    n_go    = sm.shape[1] if sm is not None else 0
    has_dtu = cfg.get('dtu_df') is not None
    ref_auprc = tctx.get('reference_auprc')

    n_high = int((sm > thr).any(axis=1).sum()) if sm is not None else 0
    high_pct = 100 * n_high / max(1, n_iso)

    types_arr = cfg.get('isoform_types')
    n_novel = 0
    if types_arr is not None:
        types_arr = np.array(types_arr)
        n_novel = int(np.sum(np.isin(types_arr, ['NIC', 'NNIC', 'novel', 'Novel'])))
    novel_pct = 100 * n_novel / max(1, n_iso)

    s_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    top_s3_genes: list[str] = []
    top_s1_genes: list[str] = []
    if classified_df is not None and 'scenario' in classified_df.columns:
        for s, cnt in classified_df['scenario'].value_counts().items():
            s_counts[int(s)] = int(cnt)
        if 'gene_id' in classified_df.columns and 'max_score' in classified_df.columns:
            top_s3_genes = _top_names(
                classified_df[classified_df['scenario'] == 3]
                .sort_values('max_score', ascending=False)['gene_id'], 3)
            top_s1_genes = _top_names(
                classified_df[classified_df['scenario'] == 1]
                .sort_values('max_score', ascending=False)['gene_id'], 3)

    macro_auprc = None
    if val_rep is not None:
        try:
            macro_auprc = val_rep.macro_auprc
        except Exception:
            pass

    # Top GO terms by coverage
    top_go_names: list[str] = []
    if sm is not None and n_go > 0:
        go_terms = cfg.get('go_terms', [])
        go_names = cfg.get('go_names', {})
        cover = (sm > thr).mean(axis=0)
        top_go_idx = np.argsort(cover)[::-1][:3]
        top_go_names = [go_names.get(go_terms[i], go_terms[i])[:30]
                        for i in top_go_idx if i < len(go_terms)]

    # Headline
    if high_pct >= 50:
        headline = (
            f"{tctx.get('name', tissue)} 데이터 {n_iso:,}개 아이소폼 중 {high_pct:.1f}%({n_high:,}개)가 "
            f"임계값 {thr} 이상의 고신뢰 GO 기능 예측 보유 — 커버리지 우수."
        )
    elif high_pct >= 20:
        headline = (
            f"{n_iso:,}개 아이소폼 중 {high_pct:.1f}%({n_high:,}개)가 고신뢰 GO 예측 — "
            f"임계값 {thr:.2f} 조정 시 커버리지 확대 가능."
        )
    else:
        headline = (
            f"{n_iso:,}개 아이소폼 중 {high_pct:.1f}%만 임계값 {thr} 이상 — "
            "임계값 인하 또는 GO 패널 점검 권장."
        )

    bullets: list[str] = [
        f"데이터셋: {tctx.get('name', tissue)} · {n_iso:,}개 아이소폼 × {n_go}개 GO term",
        f"고신뢰 예측 아이소폼: {n_high:,}개 ({high_pct:.1f}%) [임계값 {thr}]",
    ]
    if top_go_names:
        bullets.append(f"커버리지 상위 GO term: {' / '.join(top_go_names)}")
    if n_novel > 0:
        bullets.append(
            f"Novel 아이소폼(NIC+NNIC): {n_novel:,}개 ({novel_pct:.1f}%) — "
            "기존 DB 주석 없어 PRISM만 기능 예측 가능"
        )
    if s_counts[3] > 0:
        s3_str = f" (대표: {', '.join(top_s3_genes)})" if top_s3_genes else ""
        bullets.append(
            f"Scenario 3 신규 기능: {s_counts[3]:,}개 ({_pct(s_counts[3], n_iso)}){s3_str}"
        )
    if s_counts[1] > 0:
        s1_str = f" (대표: {', '.join(top_s1_genes)})" if top_s1_genes else ""
        bullets.append(
            f"Scenario 1 기능 스위치: {s_counts[1]:,}개 — 실험 검증 최우선{s1_str}"
        )
    if macro_auprc is not None:
        ref_str = ""
        if ref_auprc is not None:
            diff = macro_auprc - ref_auprc
            ref_str = f" (논문 기준치 {ref_auprc:.4f} 대비 {diff:+.4f})"
        bullets.append(f"Known 주석 검증 Macro AUPRC: {macro_auprc:.4f}{ref_str}")
    if not has_dtu:
        bullets.append("DTU 데이터 미포함 → Scenario 1·2 비활성 (Upload 모드에서 추가 가능)")

    # Interpretation
    interp_parts: list[str] = []
    if novel_pct > 5:
        interp_parts.append(
            f"Novel 아이소폼 비율 {novel_pct:.1f}%는 롱리드 시퀀싱이 기존 단편 시퀀싱 대비 "
            "훨씬 풍부한 전사체 다양성을 포착했음을 보여줍니다. "
            "NIC·NNIC 아이소폼은 InterPro·Pfam 도메인 기반 도구로 기능 예측이 불가능하며, "
            "ESM-2 기반 PRISM이 유일한 기능 예측 수단입니다."
        )
    key_fns = tctx.get('key_functions', [])
    if key_fns and s_counts[3] > 100:
        interp_parts.append(
            f"{tissue} 조직의 핵심 기능({', '.join(key_fns[:2])})에서 "
            f"S3 아이소폼 {s_counts[3]:,}개가 발견됐습니다. "
            "이는 세포 유형별 대안 스플라이싱이 조직 기능 다양성을 만드는 메커니즘과 일치하며, "
            "기존 UniProt 주석만으로는 포착할 수 없는 새로운 기능적 레퍼토리입니다."
        )
    if not interp_parts:
        interp_parts.append(
            "PRISM은 ESM-2 단백질 언어 모델의 서열 임베딩으로 GO 기능을 예측합니다. "
            "고신뢰 예측(Score > 임계값) 아이소폼은 실험 검증 후보 선정의 출발점이 됩니다."
        )

    caveats = [
        "PRISM score는 확률 추정치이며 실험 검증을 대체하지 않습니다.",
        f"임계값 {thr}는 논문의 S1×BISECT 교차 기준 — 데이터셋에 따라 조정 필요.",
    ]
    if not has_dtu:
        caveats.append("DTU 없이는 Scenario 1·2 판별 불가 — 기능 스위치 발견을 위해 DTU 분석 권장.")
    if n_iso > 30000:
        caveats.append("대규모 데이터셋에서 UMAP 투영은 무작위 샘플링 사용 — 전체 분포 반영 여부 확인 권장.")

    next_steps = []
    if s_counts[1] > 0 and top_s1_genes:
        next_steps.append(f"Scenario 1 최우선 후보 {', '.join(top_s1_genes)} → 타겟 탐색 → BISECT 분석")
    if s_counts[3] > 0 and top_s3_genes:
        next_steps.append(f"Scenario 3 신규 기능 후보 {', '.join(top_s3_genes)} → 개별 아이소폼 분석")
    if not has_dtu:
        next_steps.append("Upload 모드에서 DTU TSV 업로드 → Scenario 1·2 활성화")
    if macro_auprc is not None and ref_auprc is not None and macro_auprc < ref_auprc - 0.05:
        next_steps.append(f"AUPRC {macro_auprc:.4f}가 기준치({ref_auprc:.4f}) 미달 — 조직 맞춤 GO 패널 확인 권장")
    next_steps.append("Module Landscape 탭 → GO 기능 클러스터 공간 시각화")

    md = _to_markdown("QC & Overview", headline, bullets,
                      " ".join(interp_parts), caveats, next_steps)
    return dict(headline=headline, bullets=bullets,
                interpretation=" ".join(interp_parts),
                caveats=caveats, next_steps=next_steps, markdown=md)


# ── Module Landscape ──────────────────────────────────────────────────────────

def interpret_modules(
    cfg: dict,
    classified_df: Optional[pd.DataFrame] = None,
    module_data: Optional[dict] = None,
) -> dict:
    sm        = cfg.get('score_matrix')
    n_iso     = sm.shape[0] if sm is not None else 0
    n_go      = sm.shape[1] if sm is not None else 0
    tissue    = cfg.get('tissue', 'unknown')
    tctx      = _TISSUE_CONTEXT.get(tissue, {})
    n_modules = 0
    best_sil  = None

    if module_data is not None:
        n_modules = module_data.get('n_modules', 0)
        best_sil  = module_data.get('best_silhouette')

    # Isoform module breakdown
    top_module      = None
    top_module_cnt  = 0
    multi_module_genes = 0
    if classified_df is not None and 'primary_module' in classified_df.columns:
        mc = classified_df['primary_module'].value_counts()
        if len(mc) > 0:
            top_module    = mc.index[0]
            top_module_cnt = int(mc.iloc[0])
        if 'gene_id' in classified_df.columns:
            gene_mods = classified_df.groupby('gene_id')['primary_module'].nunique()
            multi_module_genes = int((gene_mods > 1).sum())

    sil_grade = ('우수 (>0.35)' if best_sil and best_sil > 0.35
                 else '양호 (0.20–0.35)' if best_sil and best_sil > 0.20
                 else '낮음 (<0.20)' if best_sil else '—')

    if n_modules > 0 and best_sil is not None:
        headline = (
            f"{n_go}개 GO term → {n_modules}개 기능 모듈 (Silhouette {best_sil:.3f}, {sil_grade}). "
            f"아이소폼 {n_iso:,}개의 기능 공간 구조 확인."
        )
    else:
        headline = f"{n_go}개 GO term 패널의 기능 모듈 구조 분석 ({n_iso:,}개 아이소폼)."

    bullets: list[str] = [f"GO 패널 {n_go}개 term × 아이소폼 {n_iso:,}개"]
    if n_modules > 0:
        bullets.append(f"기능 모듈: {n_modules}개 (Ward 계층적 클러스터링, Silhouette {best_sil:.3f} [{sil_grade}])")
    if top_module is not None:
        bullets.append(f"최대 모듈: M{top_module} ({top_module_cnt:,}개 아이소폼 — {_pct(top_module_cnt, n_iso)} 차지)")
    if multi_module_genes > 0:
        bullets.append(f"복수 모듈에 걸친 유전자: {multi_module_genes:,}개 (같은 유전자의 아이소폼들이 서로 다른 기능 모듈에 분포)")
    key_fns = tctx.get('key_functions', [])
    if key_fns:
        bullets.append(f"{tctx.get('name', tissue)} 조직 핵심 기능: {' / '.join(key_fns[:2])}")

    interpretation = (
        "기능 모듈은 Pearson 상관 기반 계층적 클러스터링으로 구성됩니다. "
        "동일 모듈의 GO term들은 같은 아이소폼 집합에서 함께 높은 스코어를 보이는 기능적으로 연관된 생물학적 과정입니다. "
        f"Silhouette {best_sil:.3f}는 모듈 간 경계가 {'명확하여 기능 특이성이 높음' if best_sil and best_sil > 0.3 else '적당함 — GO 패널 이질성 확인 권장'}을 의미합니다. "
    )
    if multi_module_genes > 0:
        interpretation += (
            f"복수 모듈에 걸친 유전자 {multi_module_genes:,}개는 단일 유전자에서 아이소폼에 따라 "
            "서로 다른 기능 모듈이 활성화될 수 있음을 시사합니다 — 기능 스위치의 잠재적 근거입니다."
        )

    caveats = [
        "모듈 구성은 GO 패널 선택에 의존합니다 — 패널이 다르면 다른 모듈 구조가 나타납니다.",
        f"클러스터 수 k={n_modules}는 Silhouette 최적화로 자동 선택됩니다 (k=2–12 탐색).",
    ]
    if best_sil is not None and best_sil < 0.15:
        caveats.append("Silhouette < 0.15 — GO term 간 상관이 낮거나 패널이 이질적일 수 있습니다.")

    next_steps = ["각 모듈의 대표 GO term 확인 → 생물학적 기능 레이블 부여"]
    if multi_module_genes > 0:
        next_steps.append(f"복수 모듈 유전자 {multi_module_genes:,}개 → 타겟 탐색에서 아이소폼 비교")
    next_steps.append(f"타겟 탐색 → 상위 모듈(M{top_module}) 아이소폼 필터링 후 S3/S1 후보 확인")

    md = _to_markdown("Module Landscape", headline, bullets,
                      interpretation, caveats, next_steps)
    return dict(headline=headline, bullets=bullets, interpretation=interpretation,
                caveats=caveats, next_steps=next_steps, markdown=md)


# ── Functional Patterns ───────────────────────────────────────────────────────

def interpret_functional_patterns(
    cfg: dict,
    classified_df: Optional[pd.DataFrame] = None,
) -> dict:
    sm     = cfg.get('score_matrix')
    n_iso  = sm.shape[0] if sm is not None else 0
    thr    = cfg.get('score_threshold', 0.4)
    tissue = cfg.get('tissue', 'unknown')

    types_arr = np.array(cfg.get('isoform_types') or [])
    known_mean = nic_mean = nnic_mean = None
    n_known = n_nic = n_nnic = 0
    if sm is not None and len(types_arr) == sm.shape[0]:
        for t, name in [('known','known'), ('NIC','nic'), ('NNIC','nnic')]:
            mask = types_arr == t
            cnt = int(mask.sum())
            if t == 'known':   n_known = cnt
            elif t == 'NIC':   n_nic   = cnt
            else:              n_nnic  = cnt
            if cnt > 0:
                val = float(sm[mask].mean())
                if t == 'known':   known_mean = val
                elif t == 'NIC':   nic_mean   = val
                else:              nnic_mean  = val

    # Top GO terms by mean score
    go_terms = cfg.get('go_terms', [])
    go_names = cfg.get('go_names', {})
    top_go_names: list[str] = []
    if sm is not None and len(go_terms) == sm.shape[1]:
        mean_by_go = sm.mean(axis=0)
        top_idx = np.argsort(mean_by_go)[::-1][:3]
        top_go_names = [go_names.get(go_terms[i], go_terms[i])[:28] for i in top_idx]

    # Multi-functional isoforms
    n_multi = 0
    if classified_df is not None and 'n_high_go' in classified_df.columns:
        n_multi = int((classified_df['n_high_go'] > 1).sum())

    headline_parts = []
    if known_mean and nic_mean:
        diff = nic_mean - known_mean
        direction = "높음" if diff > 0.01 else "낮음" if diff < -0.01 else "유사"
        headline_parts.append(f"NIC 평균 스코어 {nic_mean:.3f}는 Known 대비 {diff:+.3f} ({direction})")
    if top_go_names:
        headline_parts.append(f"최고 예측 기능: {top_go_names[0]}")
    headline = " / ".join(headline_parts) if headline_parts else f"{n_iso:,}개 아이소폼 기능 패턴 분석."

    bullets: list[str] = [
        f"데이터셋: {n_iso:,}개 아이소폼 (임계값 {thr})",
    ]
    if n_known or n_nic or n_nnic:
        bullets.append(f"Known: {n_known:,}개 / NIC: {n_nic:,}개 / NNIC: {n_nnic:,}개")
    if known_mean is not None and nic_mean is not None:
        bullets.append(f"평균 PRISM 스코어: Known {known_mean:.3f} / NIC {nic_mean:.3f}"
                       + (f" / NNIC {nnic_mean:.3f}" if nnic_mean else ""))
    if top_go_names:
        bullets.append(f"평균 스코어 상위 GO term: {' / '.join(top_go_names)}")
    if n_multi > 0:
        bullets.append(f"다기능 아이소폼(GO ≥2개 고신뢰): {n_multi:,}개 ({_pct(n_multi, n_iso)})")

    interp = (
        "히트맵의 밝은 셀은 해당 아이소폼 타입이 그 GO 기능을 높은 점수로 예측받는 것을 의미합니다. "
        "NIC·NNIC Novel 아이소폼이 Known과 다른 GO term에서 높은 점수를 보인다면 "
        "구조적 차이가 기능적 다양성으로 이어지는 생물학적 증거입니다. "
    )
    if known_mean and nic_mean:
        diff = nic_mean - known_mean
        if abs(diff) > 0.02:
            dir_txt = "높은" if diff > 0 else "낮은"
            interp += (f"NIC 아이소폼의 평균 스코어가 Known 대비 {diff:+.3f} {dir_txt} 것은 "
                       "대안 스플라이싱이 기능적으로 의미 있는 새로운 전사체를 생성하고 있음을 시사합니다.")

    caveats = [
        "타입 레이블 없음(None/unknown) 시 별도 그룹으로 표시됩니다.",
        "평균 스코어는 데이터셋 내 아이소폼 구성에 따라 달라집니다.",
    ]
    next_steps = []
    if top_go_names:
        next_steps.append(f"{top_go_names[0]} GO term 고신뢰 아이소폼 목록 → 타겟 탐색")
    if n_multi > 0:
        next_steps.append(f"다기능 아이소폼 {n_multi:,}개 → 개별 아이소폼 분석에서 기능 프로파일 확인")
    next_steps.append("Novel 타입만 높은 스코어를 보이는 GO term → S3 후보로 분류")

    md = _to_markdown("Functional Patterns", headline, bullets,
                      interp, caveats, next_steps)
    return dict(headline=headline, bullets=bullets, interpretation=interp,
                caveats=caveats, next_steps=next_steps, markdown=md)


# ── Condition Analysis ────────────────────────────────────────────────────────

def interpret_condition(
    cfg: dict,
    dtu_df: Optional[pd.DataFrame] = None,
    gain: int = 0,
    loss: int = 0,
    neutral: int = 0,
    enrichment_df: Optional[pd.DataFrame] = None,
    conseq_df: Optional[pd.DataFrame] = None,
) -> dict:
    total    = gain + loss + neutral
    has_dtu  = dtu_df is not None and total > 0
    tissue   = cfg.get('tissue', 'unknown')
    tctx     = _TISSUE_CONTEXT.get(tissue, {})
    n_go     = len(cfg.get('go_terms', []))

    if not has_dtu:
        headline = "DTU 데이터 없음 — 조건 비교 분석을 위해 DTU 파일을 업로드하세요."
        bullets  = [
            "DTU(Differential Transcript Usage)는 두 조건 간 아이소폼 사용 비율 변화를 검정합니다.",
            "satuRn · DEXSeq · IsoformSwitchAnalyzeR 출력 포맷 지원",
            f"현재 GO 패널 {n_go}개 term — DTU 업로드 시 즉시 GAIN/LOSS 매트릭스 생성",
        ]
        interp = (
            "DTU 결과와 PRISM 기능 예측을 결합하면 '어떤 아이소폼이 어떤 조건에서 "
            "어떤 기능을 얼마나 수행하는지' 정량화할 수 있습니다."
        )
        caveats = ["DTU 없이는 Scenario 1·2 판별 불가."]
        next_steps = ["Upload 모드에서 DTU TSV 업로드 후 재분석"]
        md = _to_markdown("Condition Analysis", headline, bullets, interp, caveats, next_steps)
        return dict(headline=headline, bullets=bullets, interpretation=interp,
                    caveats=caveats, next_steps=next_steps, markdown=md)

    gain_pct = 100 * gain / max(1, total)
    loss_pct = 100 * loss / max(1, total)
    neut_pct = 100 * neutral / max(1, total)

    # Specific gene names from conseq_df
    top_gain_genes: list[str] = []
    top_loss_genes: list[str] = []
    top_gain_go    = ""
    top_loss_go    = ""
    n_gain_genes   = 0
    n_loss_genes   = 0
    n_mixed_genes  = 0

    if conseq_df is not None and not conseq_df.empty:
        gain_sub = conseq_df[conseq_df['consequence'] == 'GAIN']
        loss_sub = conseq_df[conseq_df['consequence'] == 'LOSS']
        n_gain_genes = int(gain_sub['gene_id'].nunique())
        n_loss_genes = int(loss_sub['gene_id'].nunique())
        mixed = conseq_df[conseq_df['consequence'].isin(['GAIN','LOSS'])].groupby('gene_id')['consequence'].nunique()
        n_mixed_genes = int((mixed > 1).sum())

        if not gain_sub.empty:
            top_g = (gain_sub.groupby('gene_id')['score_delta'].max()
                     .nlargest(3).index.tolist())
            top_gain_genes = [str(g) for g in top_g]
            best_gain_row = gain_sub.loc[gain_sub['score_delta'].idxmax()]
            top_gain_go = str(best_gain_row.get('go_name', ''))[:35]
        if not loss_sub.empty:
            top_l = (loss_sub.groupby('gene_id')['score_delta'].min()
                     .nsmallest(3).index.tolist())
            top_loss_genes = [str(g) for g in top_l]
            best_loss_row = loss_sub.loc[loss_sub['score_delta'].idxmin()]
            top_loss_go = str(best_loss_row.get('go_name', ''))[:35]

    dominant = ("GAIN 우세" if gain > loss * 1.5
                else "LOSS 우세" if loss > gain * 1.5
                else "GAIN/LOSS 균형")

    if n_gain_genes > 0 or n_loss_genes > 0:
        headline = (
            f"DTU 분석: GAIN {n_gain_genes:,}개 유전자 ({gain_pct:.1f}%), "
            f"LOSS {n_loss_genes:,}개 유전자 ({loss_pct:.1f}%) — {dominant}."
        )
    else:
        headline = (
            f"DTU 이벤트 {total:,}개 중 GAIN {gain:,}개 ({gain_pct:.1f}%), "
            f"LOSS {loss:,}개 ({loss_pct:.1f}%) — {dominant}."
        )

    bullets: list[str] = []
    if n_gain_genes:
        gain_str = f"대표: {', '.join(top_gain_genes)}" if top_gain_genes else ""
        bullets.append(f"🔺 GAIN 유전자: {n_gain_genes:,}개 — 최고 기능 '{top_gain_go}' ({gain_str})")
    if n_loss_genes:
        loss_str = f"대표: {', '.join(top_loss_genes)}" if top_loss_genes else ""
        bullets.append(f"🔻 LOSS 유전자: {n_loss_genes:,}개 — 최고 기능 '{top_loss_go}' ({loss_str})")
    if n_mixed_genes:
        bullets.append(f"기능 재배치(GAIN+LOSS 동시): {n_mixed_genes:,}개 유전자 — 같은 유전자에서 GO term별 방향 상이")
    bullets.append(f"NEUTRAL (기능 변화 없음): {neutral:,}개 이벤트 ({neut_pct:.1f}%)")
    bullets.append(f"분석 범위: {n_go}개 GO term × DTU 유전자 집합")

    # Enrichment
    sig_enriched: list[str] = []
    if enrichment_df is not None and not enrichment_df.empty:
        sig_col = 'FDR' if 'FDR' in enrichment_df.columns else 'pvalue'
        sig_df = enrichment_df[enrichment_df[sig_col] < 0.05] if sig_col in enrichment_df.columns else pd.DataFrame()
        if not sig_df.empty:
            go_col = 'GO_term' if 'GO_term' in sig_df.columns else 'go_name'
            sig_enriched = sig_df[go_col].tolist()[:3] if go_col in sig_df.columns else []
            bullets.append(f"GO Enrichment 유의 term: {len(sig_df)}개 — 대표: {', '.join(sig_enriched[:2])}")

    # Biological interpretation
    tname = tctx.get('name', tissue)
    if gain > loss * 1.5:
        interp = (
            f"{tname} 데이터에서 GAIN이 우세합니다({gain_pct:.1f}%). "
            "질병 조건에서 기능적으로 더 활성화된 아이소폼이 선택됨을 시사합니다. "
        )
        if top_gain_genes:
            interp += (
                f"대표 GAIN 유전자 {', '.join(top_gain_genes[:2])}에서 "
                f"'{top_gain_go}' 기능이 새롭게 획득되는 것은 "
                "새로운 기능 획득이 병리 기전 또는 보상 반응일 가능성이 있습니다."
            )
    elif loss > gain * 1.5:
        interp = (
            f"{tname} 데이터에서 LOSS가 우세합니다({loss_pct:.1f}%). "
            "질병 조건에서 기존 기능을 가진 아이소폼이 대체되는 패턴입니다. "
        )
        if top_loss_genes:
            interp += (
                f"대표 LOSS 유전자 {', '.join(top_loss_genes[:2])}에서 "
                f"'{top_loss_go}' 기능 손실은 "
                "해당 기능의 loss-of-function이 병리 기전과 연관될 수 있습니다."
            )
    else:
        interp = (
            f"GAIN({gain_pct:.1f}%)과 LOSS({loss_pct:.1f}%)가 균형 잡힌 분포입니다. "
            "조건 변화에 따라 기능의 재배치(functional remodeling)가 일어나고 있을 가능성이 있으며, "
            f"혼합 패턴 유전자 {n_mixed_genes:,}개에서 아이소폼별 기능 역할 분담이 관찰됩니다."
        )
    if sig_enriched:
        interp += f" GO Enrichment에서 '{sig_enriched[0]}' 등이 유의미하게 enriched됩니다."

    caveats = [
        "GAIN/LOSS는 PRISM 스코어 차이(delta > 임계값) 기반 — 임계값 조정 시 분류 결과 달라집니다.",
        "DTU 유의성(p-value, FDR) 기준에 따라 포함 케이스가 달라집니다.",
        "기능 차이 해석은 PRISM 예측 신뢰도에 의존합니다 — 실험 검증 필요.",
    ]

    next_steps = []
    if top_gain_genes:
        next_steps.append(f"GAIN 대표 유전자 {', '.join(top_gain_genes[:2])} → 개별 아이소폼 분석 → BISECT")
    if top_loss_genes:
        next_steps.append(f"LOSS 대표 유전자 {', '.join(top_loss_genes[:2])} → 기능 손실 아이소폼 구조 확인")
    next_steps.append("GO Enrichment 탭에서 GAIN/LOSS에 enriched된 생물학적 과정 확인")
    if n_mixed_genes > 5:
        next_steps.append(f"기능 재배치 유전자 {n_mixed_genes:,}개 → 모듈별 GAIN/LOSS 패턴 드릴다운")

    md = _to_markdown("Condition Analysis", headline, bullets, interp, caveats, next_steps)
    return dict(headline=headline, bullets=bullets, interpretation=interp,
                caveats=caveats, next_steps=next_steps, markdown=md)


# ── Target Hub (per-gene) ─────────────────────────────────────────────────────

def interpret_target_gene(
    gene: str,
    classified_df: Optional[pd.DataFrame] = None,
    dtu_df: Optional[pd.DataFrame] = None,
    module_data: Optional[dict] = None,
    cfg: Optional[dict] = None,
) -> dict:
    if classified_df is None or len(classified_df) == 0:
        headline = f"{gene}: 분석 데이터 없음."
        md = _to_markdown(f"타겟 — {gene}", headline, [], "", [], [])
        return dict(headline=headline, bullets=[], interpretation="",
                    caveats=[], next_steps=[], markdown=md)

    go_names = (cfg or {}).get('go_names', {})
    thr      = (cfg or {}).get('score_threshold', 0.4)

    gene_df = (classified_df[classified_df['gene_id'] == gene]
               if 'gene_id' in classified_df.columns
               else classified_df[classified_df['isoform_id'].str.startswith(gene)])

    n_iso    = len(gene_df)
    max_score = float(gene_df['max_score'].max()) if 'max_score' in gene_df.columns and n_iso > 0 else 0.0
    top_go    = str(gene_df.loc[gene_df['max_score'].idxmax(), 'max_go']
                    if 'max_go' in gene_df.columns and n_iso > 0 else '—')
    top_go_full = go_names.get(top_go, top_go)[:40]
    n_high   = int((gene_df['n_high_go'] > 0).sum()) if 'n_high_go' in gene_df.columns else 0
    scenario = int(gene_df['scenario'].mode()[0]) if 'scenario' in gene_df.columns and n_iso > 0 else 4

    # Score statistics vs dataset
    dataset_mean = float(classified_df['max_score'].mean()) if 'max_score' in classified_df.columns else 0
    gene_vs_dataset = max_score - dataset_mean

    # DTU info
    dtu_delta = None
    dtu_event_count = 0
    if dtu_df is not None and 'isoform_id' in dtu_df.columns and 'isoform_id' in gene_df.columns:
        gene_dtu = dtu_df[dtu_df['isoform_id'].isin(gene_df['isoform_id'])]
        dtu_event_count = len(gene_dtu)
        if dtu_event_count > 0 and 'dtu_delta_if' in gene_dtu.columns:
            dtu_delta = float(gene_dtu['dtu_delta_if'].abs().max())

    # Score range across isoforms
    score_range = (float(gene_df['max_score'].max() - gene_df['max_score'].min())
                   if 'max_score' in gene_df.columns and n_iso > 1 else 0.0)

    # Module info
    mod_ids: list[str] = []
    if 'primary_module' in gene_df.columns:
        mod_ids = [f"M{int(m)}" for m in gene_df['primary_module'].dropna().unique()[:3]]

    s_labels = {1: "S1 기능 스위치 ⭐⭐⭐", 2: "S2 발현 스위치 ⭐⭐",
                3: "S3 신규 기능 ⭐⭐⭐", 4: "S4 배경"}
    s_label = s_labels.get(scenario, f"S{scenario}")

    headline = (
        f"{gene}: {n_iso}개 아이소폼, 최고 스코어 {max_score:.3f} at {top_go_full[:25]}, "
        f"{s_label} (데이터셋 평균 {dataset_mean:.3f} 대비 {gene_vs_dataset:+.3f})."
    )

    bullets: list[str] = [
        f"아이소폼 수: {n_iso}개 (고신뢰 예측: {n_high}개, {_pct(n_high, n_iso)})",
        f"최고 PRISM 스코어: {max_score:.3f} — GO: {top_go_full}",
        f"아이소폼 간 스코어 분산: {score_range:.3f} ({'기능 특이성 높음' if score_range > 0.3 else '균질한 기능 프로파일'})",
        f"데이터셋 내 순위: 평균 {dataset_mean:.3f} 대비 {gene_vs_dataset:+.3f}",
    ]
    if dtu_event_count > 0:
        dtu_str = f" (최대 ΔIF={dtu_delta:.3f})" if dtu_delta else ""
        bullets.append(f"DTU 이벤트: {dtu_event_count}개{dtu_str}")
    else:
        bullets.append("DTU 이벤트 없음 (또는 DTU 데이터 미포함)")
    if mod_ids:
        bullets.append(f"기능 모듈: {', '.join(mod_ids)}")

    if scenario == 1:
        interp = (
            f"{gene}는 DTU와 신규 GO 예측이 동시에 확인된 Scenario 1 최우선 후보입니다. "
            f"'{top_go_full}' 기능이 조건에 따라 달라질 가능성이 높으며, "
            "BISECT 파이프라인(구조·도메인·PPI 통합 분석)이 권장됩니다."
        )
    elif scenario == 3:
        interp = (
            f"{gene}는 기존 Ensembl 주석과 다른 '{top_go_full}' 기능이 예측된 S3 후보입니다. "
            f"스코어 {max_score:.3f}로 데이터셋 평균({dataset_mean:.3f}) 대비 "
            f"{gene_vs_dataset:+.3f}만큼 높습니다. "
            "DTU와 무관하게 항상 발현되며 새로운 기능이 확인됐을 가능성이 있습니다."
        )
    else:
        interp = (
            f"{gene}의 {n_iso}개 아이소폼에서 최고 스코어 {max_score:.3f} "
            f"('{top_go_full}')가 확인됩니다. "
            f"아이소폼 간 스코어 분산 {score_range:.3f}은 "
            f"{'기능 특이성이 높아 개별 아이소폼 분석 가치가 있습니다' if score_range > 0.3 else '기능 프로파일이 균질합니다'}."
        )

    caveats = [
        "Quick Card는 데이터셋 내 발현 아이소폼만 반영합니다.",
        "시나리오 분류는 입력 DTU 데이터 품질에 의존합니다.",
    ]
    next_steps = [f"개별 아이소폼 분석 페이지에서 {gene} 아이소폼 상세 확인"]
    if scenario in [1, 3]:
        next_steps.insert(0, "BISECT 파이프라인으로 구조·도메인·PPI 통합 분석")
    if score_range > 0.3:
        next_steps.append(f"스코어 분산 {score_range:.3f} — Within-gene 비교 차트로 기능 분기 확인")

    md = _to_markdown(f"타겟 분석 — {gene}", headline, bullets,
                      interp, caveats, next_steps)
    return dict(headline=headline, bullets=bullets, interpretation=interp,
                caveats=caveats, next_steps=next_steps, markdown=md)


# ── Individual Isoform ────────────────────────────────────────────────────────

def interpret_isoform(
    isoform_id: str,
    classified_df: Optional[pd.DataFrame] = None,
    module_data: Optional[dict] = None,
    cfg: Optional[dict] = None,
) -> dict:
    go_names = (cfg or {}).get('go_names', {})
    row = None
    if classified_df is not None and 'isoform_id' in classified_df.columns:
        matches = classified_df[classified_df['isoform_id'] == isoform_id]
        if len(matches) > 0:
            row = matches.iloc[0]

    if row is None:
        headline = f"{isoform_id}: 분석 데이터 없음."
        md = _to_markdown(f"아이소폼 — {isoform_id}", headline, [], "", [], [])
        return dict(headline=headline, bullets=[], interpretation="",
                    caveats=[], next_steps=[], markdown=md)

    max_score  = float(row.get('max_score', 0))
    max_go     = str(row.get('max_go', '—'))
    max_go_full = go_names.get(max_go, max_go)[:40]
    n_high_go  = int(row.get('n_high_go', 0))
    scenario   = int(row.get('scenario', 4))
    iso_type   = str(row.get('isoform_type', row.get('type', '—')))
    gene_id    = str(row.get('gene_id', '—'))
    dtu_flag   = bool(row.get('dtu_flag', False))
    novel_flag = bool(row.get('novel_go_flag', False))
    dtu_delta  = float(row.get('dtu_delta_if', 0)) if 'dtu_delta_if' in row else None
    module_id  = row.get('primary_module')

    # Dataset quantile position
    dataset_mean = (float(classified_df['max_score'].mean())
                    if classified_df is not None and 'max_score' in classified_df.columns else None)
    quantile_pct = None
    if classified_df is not None and 'max_score' in classified_df.columns:
        q = float((classified_df['max_score'] < max_score).mean() * 100)
        quantile_pct = q

    s_labels = {1: "S1 기능 스위치 ⭐⭐⭐", 2: "S2 발현 스위치 ⭐⭐",
                3: "S3 신규 기능 ⭐⭐⭐", 4: "S4 배경"}
    s_label = s_labels.get(scenario, f"S{scenario}")

    headline = (
        f"{isoform_id} ({gene_id}, {iso_type}): 스코어 {max_score:.3f} at {max_go_full[:25]}, "
        f"{s_label}" +
        (f", 데이터셋 내 상위 {100-quantile_pct:.0f}% 수준." if quantile_pct is not None else ".")
    )

    bullets: list[str] = [
        f"유전자: {gene_id} / 타입: {iso_type}",
        f"최고 PRISM 스코어: {max_score:.3f} ({max_go_full})",
        f"고신뢰 GO term 수: {n_high_go}개",
        f"시나리오: {s_label}",
    ]
    if quantile_pct is not None and dataset_mean is not None:
        bullets.append(f"데이터셋 내 위치: 상위 {100-quantile_pct:.0f}% (평균 {dataset_mean:.3f} 대비 {max_score-dataset_mean:+.3f})")
    if dtu_flag:
        delta_str = f" (ΔIF={dtu_delta:.3f})" if dtu_delta is not None else ""
        bullets.append(f"DTU 검출됨{delta_str}")
    if module_id is not None:
        bullets.append(f"기능 모듈: M{int(module_id)}")

    if scenario == 1:
        interp = (
            f"{isoform_id}는 DTU 이벤트({f'ΔIF={dtu_delta:.3f}' if dtu_delta else 'p < 0.05'})와 "
            f"신규 GO 기능('{max_go_full}', score={max_score:.3f})이 동시에 확인된 "
            "Scenario 1 최우선 후보입니다. "
            "BISECT 파이프라인 실행으로 AlphaFold 구조, InterPro 도메인, STRING PPI 통합 분석을 권장합니다."
        )
    elif scenario == 3:
        if iso_type in ['NIC', 'NNIC', 'novel']:
            interp = (
                f"Novel 아이소폼({iso_type}) {isoform_id}에서 기존 Ensembl 주석에 없는 "
                f"GO 기능('{max_go_full}')이 예측됩니다 (score={max_score:.3f}). "
                f"이 아이소폼은 UniProt·InterPro 도구로는 분석 불가능하며, "
                "PRISM ESM-2 서열 특징으로만 기능 추론이 가능합니다."
            )
        else:
            interp = (
                f"{isoform_id}는 기존 Ensembl 등록 아이소폼이지만 "
                f"주석 외 GO term '{max_go_full}' (score={max_score:.3f})이 예측됩니다. "
                "도메인 재배치 또는 신규 exon 조합이 기능 변화를 일으킬 가능성이 있습니다."
            )
    else:
        interp = (
            f"{isoform_id}는 '{max_go_full}' 기능에서 최고 스코어 {max_score:.3f}를 보입니다"
            + (f" (데이터셋 내 상위 {100-quantile_pct:.0f}% 수준)." if quantile_pct else ".") +
            f" 고신뢰 GO term {n_high_go}개 — "
            "같은 유전자의 다른 아이소폼과 비교해 기능 특이성을 확인하세요."
        )

    caveats = [
        "아이소폼 수준 예측은 ESM-2 임베딩 기반이며 실험 검증 필요.",
        "NNIC 타입은 스플라이스 사이트 예측 오류를 포함할 수 있습니다.",
    ]
    if max_score < 0.4:
        caveats.append(f"최고 스코어 {max_score:.3f}가 임계값 미만 — 예측 신뢰도 낮음.")

    next_steps = ["BISECT 파이프라인으로 구조·도메인·PPI 통합 분석"]
    if scenario in [1, 3]:
        next_steps.insert(0, f"같은 유전자({gene_id}) 내 다른 아이소폼과 Within-gene 비교")
    if n_high_go > 1:
        next_steps.append(f"고신뢰 GO {n_high_go}개 전체 목록에서 기능 패턴 분석")

    md = _to_markdown(f"아이소폼 — {isoform_id}", headline, bullets,
                      interp, caveats, next_steps)
    return dict(headline=headline, bullets=bullets, interpretation=interp,
                caveats=caveats, next_steps=next_steps, markdown=md)


# ── BISECT Case ───────────────────────────────────────────────────────────────

def interpret_bisect(case_result: dict) -> dict:
    isoform_id  = case_result.get('isoform_id', '—')
    gene        = case_result.get('gene', isoform_id.split('-')[0] if '-' in isoform_id else isoform_id)
    tier        = case_result.get('tier', '—')
    n_pass      = case_result.get('n_modules_pass', 0)
    n_total     = case_result.get('n_modules_total', 15)
    bisect_score = case_result.get('bisect_score')
    top_evidence = case_result.get('evidence', case_result.get('modules_passed', []))
    cell_type    = case_result.get('cell_type', '—')
    domain_change = case_result.get('domain_change', False)
    pass_pct     = 100 * n_pass / max(1, n_total)

    # brain_672 PRISM scores (added by update_bisect_prism_scores.py)
    ct_top_go    = case_result.get('prism_ct_top_go', [])   # list of {go_id, go_name, score}
    ad_top_go    = case_result.get('prism_ad_top_go', [])
    gain_go      = case_result.get('prism_gain_go', [])      # AD > CT
    loss_go      = case_result.get('prism_loss_go', [])      # CT > AD
    ct_max_score = case_result.get('prism_ct_max_score')
    ad_max_score = case_result.get('prism_ad_max_score')
    ct_max_go    = case_result.get('prism_ct_max_go', '—')
    ad_max_go    = case_result.get('prism_ad_max_go', '—')
    match_ct     = case_result.get('prism_match_ct', 'unknown')
    match_ad     = case_result.get('prism_match_ad', 'unknown')

    # Fallback to legacy single-score fields
    prism_score  = ad_max_score or case_result.get('prism_max_score', case_result.get('top_prism_score'))
    prism_go     = ad_max_go or case_result.get('prism_max_go', case_result.get('top_go', '—'))

    tier_labels = {'A': "Tier A — 최우선 실험 검증 후보", 'S': "Tier S — 최고 우선순위",
                   'B': "Tier B — 유망 후보 (추가 증거 권장)", 'C': "Tier C — 증거 부족"}
    tier_label = tier_labels.get(str(tier), f"Tier {tier}")

    # Headline: lead with functional switch if PRISM data available
    if gain_go and loss_go:
        headline = (
            f"{gene} ({cell_type}): BISECT {n_pass}/{n_total} PASS, {tier_label} | "
            f"CT 핵심: '{ct_max_go[:30]}' → AD 핵심: '{ad_max_go[:30]}'"
            + (" · 도메인 변화 ✅" if domain_change else "") + "."
        )
    else:
        headline = (
            f"{isoform_id} ({gene}, {cell_type}): BISECT {n_pass}/{n_total} 모듈 PASS ({pass_pct:.0f}%), "
            f"{tier_label}" + (" · 도메인 변화 확인 ✅" if domain_change else "") + "."
        )

    bullets: list[str] = [
        f"BISECT 통과 모듈: {n_pass}/{n_total}개 ({pass_pct:.0f}%)",
        f"등급: {tier_label}",
    ]
    # CT → AD functional transition summary
    if ct_top_go and ad_top_go:
        ct_names = ', '.join(x['go_name'][:25] for x in ct_top_go[:3])
        ad_names = ', '.join(x['go_name'][:25] for x in ad_top_go[:3])
        bullets.append(f"CT 주요 기능 (brain_672): {ct_names}")
        bullets.append(f"AD 주요 기능 (brain_672): {ad_names}")
    if gain_go:
        top_gain = gain_go[0]
        bullets.append(f"🔺 기능 획득 (GAIN) top: '{top_gain['go_name'][:35]}' (Δ={top_gain['delta']:+.3f})")
    if loss_go:
        top_loss = loss_go[0]
        bullets.append(f"🔻 기능 손실 (LOSS) top: '{top_loss['go_name'][:35]}' (Δ={top_loss['delta']:+.3f})")
    if prism_score is not None:
        match_note = "" if match_ad == 'exact' else " (gene_median)"
        bullets.append(f"AD PRISM 최고 스코어: {float(prism_score):.3f} at {str(prism_go)[:35]}{match_note}")
    if bisect_score is not None:
        bullets.append(f"BISECT 통합 점수: {float(bisect_score):.3f}")
    if domain_change:
        bullets.append("✅ 도메인 구조 변화 확인 — 단백질 기능 변화 직접 증거")
    if top_evidence:
        bullets.append(f"통과 모듈: {', '.join(str(e) for e in top_evidence[:5])}")

    # Biological interpretation — integrate PRISM GO term evidence
    gain_str = (f"'{gain_go[0]['go_name'][:35]}' (Δ={gain_go[0]['delta']:+.3f})" if gain_go else None)
    loss_str = (f"'{loss_go[0]['go_name'][:35]}' (Δ={loss_go[0]['delta']:+.3f})" if loss_go else None)

    if n_pass >= 10:
        interp = (
            f"{gene} ({cell_type})는 BISECT 15개 모듈 중 {n_pass}개({pass_pct:.0f}%)에서 양성 증거를 보이는 "
            "강력한 기능 스위치 후보입니다. "
            "구조(AlphaFold 안정성)·도메인(InterPro 재배치)·PPI(STRING) 다층 증거가 수렴합니다."
        )
        if gain_str and loss_str:
            interp += (
                f" PRISM (brain_672, 672 GO term) 분석에서 CT 아이소폼은 '{ct_max_go[:30]}' 기능이 "
                f"핵심이었으나, AD 전환 후 {gain_str} 기능이 새롭게 획득되고 "
                f"{loss_str} 기능이 손실됩니다."
            )
        if domain_change:
            interp += " 도메인 구조 변화가 확인되어 기능 변화의 직접적 근거가 됩니다."
    elif n_pass >= 6:
        interp = (
            f"{gene}는 BISECT {n_pass}개 모듈 PASS — 유망하지만 일부 증거가 불확실합니다. "
        )
        if gain_str or loss_str:
            parts = []
            if gain_str: parts.append(f"기능 획득: {gain_str}")
            if loss_str: parts.append(f"기능 손실: {loss_str}")
            interp += "PRISM (brain_672) 기반 " + "; ".join(parts) + "."
        interp += " 핵심 모듈(구조·도메인·발현)에서 음성인 경우 추가 실험 설계가 필요합니다."
    else:
        interp = (
            f"{gene}는 BISECT {n_pass}개 모듈만 통과 — 현재 공개 DB 기반 증거로는 "
            "기능 스위치 가설 뒷받침이 어렵습니다."
        )
        if gain_str:
            interp += f" PRISM (brain_672) 기준 {gain_str} 기능 획득이 예측되므로 추가 실험 설계를 고려하세요."

    caveats = [
        "BISECT는 공개 DB(AlphaFold, STRING, Pfam) 기반 — DB 버전에 따라 결과 달라질 수 있음.",
        f"Tier {tier} 판정은 {n_pass}/{n_total} 모듈 통과 기준이며 문헌 검증 필요.",
    ]
    if match_ct != 'exact' or match_ad != 'exact':
        caveats.append(
            f"PRISM 스코어 매칭: CT={match_ct}, AD={match_ad}. "
            "'gene_median'은 해당 유전자 아이소폼 중앙값으로 대체된 추정치입니다."
        )

    next_steps = [
        f"개별 아이소폼 분석 페이지에서 {gene} PRISM 스코어 상세 확인",
        "실험 검증: RT-PCR로 조건별 아이소폼 비율 확인",
        "단백질 기능 검증: 과발현 / 녹다운 시스템 구축",
    ]
    if gain_go:
        next_steps.insert(1, f"획득 기능 '{gain_go[0]['go_name'][:30]}' → 관련 단백질 결합/활성 검증")
    if loss_go:
        next_steps.insert(2, f"손실 기능 '{loss_go[0]['go_name'][:30]}' → CT 아이소폼 재발현 rescue 실험")
    if tier in ['A', 'S']:
        next_steps.insert(0, "논문 주요 케이스 등재 검토 — 그림 작성 우선 순위 ⭐")
    if domain_change:
        next_steps.insert(1, "AlphaFold 구조 비교 → 기능 도메인 손실/획득 확인")

    md = _to_markdown(f"BISECT — {gene}", headline, bullets,
                      interp, caveats, next_steps)
    return dict(headline=headline, bullets=bullets, interpretation=interp,
                caveats=caveats, next_steps=next_steps, markdown=md)


# ── Landscape (UMAP-level) ────────────────────────────────────────────────────

def interpret_landscape(
    cfg: dict,
    classified_df: Optional[pd.DataFrame] = None,
) -> dict:
    sm     = cfg.get('score_matrix')
    n_iso  = sm.shape[0] if sm is not None else 0
    tissue = cfg.get('tissue', 'unknown')
    tctx   = _TISSUE_CONTEXT.get(tissue, {})

    n_clusters = 0
    largest_cluster = None
    if classified_df is not None and 'primary_module' in classified_df.columns:
        n_clusters = classified_df['primary_module'].nunique()
        mc = classified_df['primary_module'].value_counts()
        if len(mc):
            largest_cluster = (mc.index[0], int(mc.iloc[0]))

    s3_count = 0
    if classified_df is not None and 'scenario' in classified_df.columns:
        s3_count = int((classified_df['scenario'] == 3).sum())

    headline = (
        f"GO 스코어 공간 UMAP: {n_iso:,}개 아이소폼이 "
        f"{n_clusters}개 기능 클러스터로 구분됨 ({tctx.get('name', tissue)})."
        if n_clusters else
        f"GO 스코어 공간 UMAP: {n_iso:,}개 아이소폼 투영 ({tctx.get('name', tissue)})."
    )

    bullets: list[str] = [f"아이소폼 {n_iso:,}개 × GO 스코어 벡터 → 2D UMAP 투영"]
    if n_clusters:
        bullets.append(f"기능 클러스터: {n_clusters}개")
    if largest_cluster:
        bullets.append(f"최대 클러스터: M{largest_cluster[0]} ({largest_cluster[1]:,}개 아이소폼)")
    if s3_count:
        bullets.append(f"S3 신규 기능 아이소폼 {s3_count:,}개 — UMAP 특정 클러스터 집중 여부 확인 권장")
    key_fns = tctx.get('key_functions', [])
    if key_fns:
        bullets.append(f"핵심 기능 클러스터 확인 대상: {' / '.join(key_fns[:2])}")

    interp = (
        "UMAP은 고차원 GO 스코어 벡터를 2D로 압축합니다. "
        "가까운 점들은 기능적으로 유사한 아이소폼입니다. "
        "뚜렷한 클러스터 경계는 대안 스플라이싱이 기능 공간을 이산화하는 생물학적 증거로 해석할 수 있습니다. "
        f"S3 아이소폼 {s3_count:,}개가 특정 클러스터에 집중됐다면 그 기능이 Novel 아이소폼에서 선택적으로 나타남을 의미합니다."
        if s3_count else
        "UMAP은 고차원 GO 스코어 벡터를 2D로 압축합니다. "
        "가까운 점들은 기능적으로 유사한 아이소폼이며, "
        "뚜렷한 클러스터 경계는 대안 스플라이싱이 기능 공간을 이산화하는 생물학적 증거입니다."
    )

    caveats = [
        "UMAP은 전역 거리를 보존하지 않습니다 — 클러스터 간 상대적 위치보다 내부 구조가 중요합니다.",
        "대규모 데이터셋에서는 속도를 위해 무작위 샘플링됩니다.",
    ]
    next_steps = [
        "색칠 기준(아이소폼 타입·시나리오·최고 GO)을 바꿔가며 클러스터 구조 파악",
        "특정 클러스터 내 아이소폼 선택 후 타겟 탐색",
    ]
    if s3_count > 0:
        next_steps.insert(0, f"S3 아이소폼 {s3_count:,}개 클러스터 위치 확인 → 신규 기능 GO term 식별")

    md = _to_markdown("Module Landscape", headline, bullets, interp, caveats, next_steps)
    return dict(headline=headline, bullets=bullets, interpretation=interp,
                caveats=caveats, next_steps=next_steps, markdown=md)
