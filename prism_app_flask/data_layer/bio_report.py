"""BISECT biological function prediction report builder.

Ports and refines the `_build_bio_report_html` narrative logic from streamlit
`07_bisect.py` for Flask. The streamlit version assembled light inline-styled HTML
directly, but here we only return a **structured dict** — rendering (dark
data-instrument theme) is handled by `static/js/bisect.js`.

The returned dict carries, per case:
    tier        PRISM functional tier {label, color}
    confidence  evidence-module-count-based confidence {label, color, n, max}
    pathway     isoform-switch causal pathway step list
    regulators  TF/ASF activity-change badge list
    context     promoter · APA · mechanism context
    domains     gained/lost domains + functional annotation
    narrative   integrated-interpretation sentence list (HTML inline <b> allowed)
"""
from __future__ import annotations

# ── Knowledge bases (ported from 07_bisect.py) ──────────────────────────────
_REGULATOR_KB: dict = {
    'STAT1':   ('TF',         True,  'Key AD neuroinflammation transcription factor; repressed in microglia / excitatory neurons (Baranzini 2020)'),
    'REST':    ('TF',         True,  'Neuroprotective transcriptional repressor; decreased expression in AD de-represses synaptic genes (Lu 2014 Cell)'),
    'CREB1':   ('TF',         True,  'Neuronal survival / LTP transcription factor; reduced phosphorylation in AD impairs memory formation (Saura 2004)'),
    'SP1':     ('TF',         True,  'Binds Tau/APP promoters directly; AD vulnerability factor (Citron 2008)'),
    'SP3':     ('TF',         True,  'SP1-antagonist transcription factor; overexpressed relative to SP1 in AD → promoter competition (Black 2001)'),
    'SRSF5':   ('ASF',        True,  'Serine/Arginine Splicing Factor 5; AD-associated splicing reprogramming (Raj 2018)'),
    'SRSF7':   ('ASF',        True,  'Regulates tau exon 10 inclusion; linked to FTLD-Tau (Jiang 1998)'),
    'RBFOX1':  ('ASF',        True,  'Brain-specific ASF; regulates neurodevelopmental / AD-vulnerable exons (Bhatt 2020)'),
    'HDAC2':   ('Epigenetic', True,  'Hyperactive histone H3K27 deacetylation in AD represses neuronal genes (Gräff 2012)'),
    'SIRT1':   ('Epigenetic', True,  'Reduced NAD+-dependent deacetylation in AD hyperactivates p53/NF-κB (Kim 2007)'),
    'KLF9':    ('TF',         False, 'Newly identified; candidate repressive transcription factor, regulates oxidative-stress response'),
    'YBX1':    ('RBP',        False, 'Y-box RNA-binding protein; regulates splicing/translation, AD role not yet established'),
    'HNRNPK':  ('ASF',        False, 'hnRNP K; regulates pre-mRNA splicing/transport, newly implicated in AD'),
    'E2F3':    ('TF',         False, 'Cell-cycle / apoptosis transcription factor; possibly linked to AD neuronal cell-cycle re-entry'),
    'SETDB2':  ('Epigenetic', False, 'H3K9me3 methyltransferase; heterochromatin repression → aberrant gene expression'),
}

_MECHANISM_KO: dict = {
    'alternative_promoter':    ('Alternative Promoter', '#a78bfa',
                                'A different promoter is activated, shifting the transcription start site. '
                                'The N-terminal structure changes, potentially altering signal peptides / membrane-binding domains.'),
    'alternative_splicing':    ('Alternative Splicing', '#35C6E8',
                                'Exon inclusion/exclusion directly changes domain composition. '
                                'Changes in ASF (SRSF, RBFOX, etc.) binding sites are the main driver.'),
    'transcriptional':         ('Transcriptional Regulation', '#C99A52',
                                'Transcript abundance is modulated by TF binding changes at the same promoter. '
                                'TF activity changes are the direct cause of the isoform-ratio shift.'),
    'epigenetic_derepression': ('Epigenetic Derepression', '#E06A80',
                                'HDAC hyperactivity or DNA methylation changes open up a previously repressed exon. '
                                'Chromatin accessibility changes reshape the splicing pattern.'),
    'intron_retention':        ('Intron Retention', '#4FC08A',
                                'Reduced splicing efficiency leaves an intron in the mature mRNA. '
                                'Elevated NMD risk; protein translation must be verified.'),
}

_DOMAIN_FUNC_MAP = {
    'Kinesin':       'microtubule-based motor activity (ATP-dependent)',
    'WD40':          'β-propeller scaffold for protein–protein interactions',
    'PDZ':           'synaptic scaffolding, C-terminal peptide binding',
    'SAM':           'oligomerization / RNA-binding (context-dependent)',
    'SH3':           'proline-rich sequence binding, signaling assembly',
    'SH2':           'phosphotyrosine binding, downstream signaling',
    'RRM':           'RNA recognition motif, post-transcriptional regulation',
    'Microtub_bd':   'direct microtubule binding and stabilization',
    'NDUS4':         'NADH:ubiquinone oxidoreductase (Complex I) assembly',
    'RVT_1':         'reverse-transcriptase / RNA-dependent DNA polymerase',
    'DUF5082':       'domain of unknown function (DUF5082)',
    'ANAPC4_WD40':   'APC/C complex scaffold, cell-cycle regulation',
    'Nup160':        'nuclear pore complex, nucleocytoplasmic transport',
    'PH':            'phosphoinositide binding, membrane recruitment',
    'Guanylate_kin': 'guanylate kinase activity, scaffolding at PSD',
    'RhoGAP':        'Rho GTPase-activating protein, cytoskeleton regulation',
    'RhoGEF':        'Rho guanine-nucleotide exchange factor',
    'Pkinase':       'serine/threonine protein kinase, signal transduction',
    'CARD':          'caspase recruitment domain, apoptosis regulation',
    'FN3':           'fibronectin type-III fold, cell adhesion',
    'EGF':           'EGF receptor binding, proliferation signaling',
    'BEACH':         'lysosome/endosome biogenesis regulation',
    'GRAM':          'membrane association with PH domain',
}

_COMPLEX1_GENES = {'NDUFS4', 'NDUFS7', 'NDUFS8'}

# PRISM functional tiers — colors tuned for the dark theme.
# (color, lucide_icon_key_or_None, label). icon=None keeps the existing plain-unicode prefix
# (↔ ↑ 〜 △ ?) in the label — those are typographic symbols, not emoji, and match the app's
# instrument-terminal glyph system (▸ ☾ ☀), so they're left as-is.
_TIER_STYLE = {
    'tier1_functional_switch': ('#7C83FF', 'microscope',    'PRISM Tier 1 · Functional Switch'),
    'tier2_functional_loss':   ('#E06A80', 'trending-down', 'PRISM Tier 2 · Functional Loss'),
    'tier2_complex_loss':      ('#C0405A', 'zap',           'PRISM Tier 2 · Complex I Collapse'),
    'tier2_partial_change':    ('#C99A52', None, '↔ PRISM Tier 2 · Partial Change'),
    'tier2_gain_no_direction': ('#C99A52', None, '↑ PRISM Tier 2 · Functional Gain'),
    'tier3_gene_median':       ('#66728A', None, '〜 Tier 3 · Representative-sequence Estimate'),
    'tier3_structural_only':   ('#66728A', None, '△ Tier 3 · Structural Evidence Only'),
    'tier3_no_match':          ('#66728A', None, '? Tier 3 · No Match'),
}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _infer_prism_tier(c: dict) -> str:
    """Estimate prism_tier from structural evidence when absent from JSON (ported from 07_bisect.py)."""
    gene  = str(c.get('gene', '') or '').upper()
    af_g  = str(c.get('af_gained_confident', '') or '').strip()
    af_l  = str(c.get('af_lost_confident',  '') or '').strip()
    dom_g = str(c.get('domains_gained',     '') or '').strip()
    dom_l = str(c.get('domains_lost',       '') or '').strip()
    if gene in _COMPLEX1_GENES:
        return 'tier2_complex_loss'
    if af_g:
        return 'tier1_functional_switch'
    if af_l:
        return 'tier2_functional_loss'
    if dom_g or dom_l:
        return 'tier2_partial_change'
    return 'tier3_structural_only'


def _domain_func(d: str) -> str:
    for k, v in _DOMAIN_FUNC_MAP.items():
        if k.lower() in d.lower():
            return v
    return 'function uncharacterised'


def build_report(c: dict, regulators: list | None = None) -> dict:
    """BISECT case dict → structured biological function prediction report."""
    ct_type = str(c.get('cell_type') or '')
    ct_tx   = str(c.get('ct_transcript_id') or '')
    ad_tx   = str(c.get('ad_transcript_id') or '')

    delta   = _num(c.get('delta'))
    dtu_p   = _num(c.get('dtu_p'))
    dg_list = [d.strip() for d in str(c.get('domains_gained') or '').split(';') if d.strip()]
    dl_list = [d.strip() for d in str(c.get('domains_lost')   or '').split(';') if d.strip()]
    ppi_v   = str(c.get('ppi_verdict') or '').strip()
    ppi_p   = str(c.get('ppi_top_partner') or '').strip()
    ppi_s   = _num(c.get('ppi_top_score'))
    phylo   = _num(c.get('cons_ad_phylop'))
    cons_c  = str(c.get('cons_ad_class') or '').strip()
    mech    = str(c.get('mechanism_type') or '').strip()
    tss_cls = str(c.get('tss_class') or '').strip()
    apa_cls = str(c.get('apa_class') or '').strip()
    tss_bp  = _num(c.get('tss_diff_bp'))
    apa_bp  = _num(c.get('tts_diff_bp'))
    ad_nmd  = c.get('ad_nmd')
    af_ad   = _num(c.get('af_ad_plddt_mean'))
    af_ct   = _num(c.get('af_ct_plddt_mean'))
    af_delta = _num(c.get('af_delta_plddt'))

    all_regs = regulators or []
    mech_info = _MECHANISM_KO.get(mech, ('', '#66728A', ''))

    # ── PRISM top / gain / loss GO (JSON pre-built) ──────────────────────────
    def _go_list(key, score_key='score', n=3):
        out = []
        for d in (c.get(key) or [])[:n]:
            s = _num(d.get(score_key))
            if s is None:
                continue
            out.append({'name': (d.get('go_name') or d.get('go_id') or '')[:40], 'score': s})
        return out

    ct_top  = _go_list('prism_ct_top_go')
    ad_top  = _go_list('prism_ad_top_go')
    gain_go = [{'name': (d.get('go_name') or '')[:40], 'score': _num(d.get('ad_score')) or 0.0}
               for d in (c.get('prism_gain_go') or [])[:3]]
    loss_go = [{'name': (d.get('go_name') or '')[:40], 'score': _num(d.get('ct_score')) or 0.0}
               for d in (c.get('prism_loss_go') or [])[:3]]

    m_ct = str(c.get('prism_match_ct') or '').strip()
    m_ad = str(c.get('prism_match_ad') or '').strip()
    prism_exact = ('exact' in m_ct or 'exact' in m_ad or m_ad.startswith('proxy:'))

    # ── Confidence (evidence-module count) ───────────────────────────────────
    _bisect_tier = str(c.get('bisect_tier') or '').strip()
    _tier_pass   = _bisect_tier in ('A-DR', 'A-BP')
    ev_count = sum([
        bool(_tier_pass or (delta and abs(delta) > 0.1 and dtu_p and dtu_p < 0.05)),
        bool(str(c.get('af_gained_confident') or '').strip() or
             str(c.get('af_lost_confident') or '').strip() or
             (af_delta and abs(af_delta) > 5)),
        bool(dg_list or dl_list),
        ppi_v == 'SUPPORTED',
        'conserved' in cons_c.lower(),
        bool(all_regs),
        bool(mech and mech != 'transcriptional'),
        tss_cls not in ('same_promoter', '', 'None'),
        apa_cls not in ('same_apa', '', 'None'),
        bool(c.get('nmd_relevant') or ad_nmd),
        bool(gain_go or loss_go) and prism_exact,
    ])
    conf_label = ['Low', 'Low', 'Moderate', 'Moderate', 'High', 'High', 'Very High',
                  'Very High', 'Very High', 'Very High', 'Very High'][min(ev_count, 10)]
    conf_color = {'Low': '#E06A80', 'Moderate': '#C99A52',
                  'High': '#4FC08A', 'Very High': '#2FA36B'}[conf_label]

    # ── Narrative notes ───────────────────────────────────────────────────────
    # 원칙: 숫자/사실은 bd-grid 한 곳에만 산다. 여기 narrative 는 "해석" 절만 남긴다 —
    # 이미 위 grid(도메인/GO/기전/regulator/AF/PPI/보존성/DTU-p)에 나온 수치를 그대로
    # 재서술하지 않는다(중복 제거). 구조는 list[str] 대신 {icon,label,text} 로 —
    # 하나의 촘촘한 문단 대신 짧은 라벨별 행으로 렌더되어 가독성을 높인다.
    notes: list[dict] = []

    if delta is not None:
        notes.append({
            'icon': 'shuffle', 'label': 'DTU',
            'text': ("Isoform composition shifts from CT- to AD-dominant usage in this locus "
                     "(see DTU panel above for the per-isoform breakdown and p-value)." if delta < 0 else
                     "The AD-associated isoform's usage share increases relative to CT "
                     "(see DTU panel above for the per-isoform breakdown and p-value)."),
        })

    if af_ad and af_ct:
        af_d = af_delta if af_delta is not None else af_ad - af_ct
        stab = ("the AD isoform forms a more stable structure than the CT isoform" if af_d > 5
                else "the CT isoform is structurally more stable, with increased disorder in the AD isoform" if af_d < -5
                else "the two isoforms have similar structural stability")
        notes.append({'icon': 'box', 'label': 'Structure', 'text': f"AlphaFold prediction: {stab}."})
    elif af_ad:
        qual = "a structurally high-confidence" if af_ad >= 70 else "a partially disordered"
        notes.append({'icon': 'box', 'label': 'Structure',
                      'text': f"AlphaFold prediction indicates the AD isoform folds into {qual} structure."})

    if ppi_v == 'SUPPORTED' and ppi_p:
        notes.append({'icon': 'link', 'label': 'PPI',
                      'text': f"STRING predicts a new interaction with <b>{ppi_p}</b>, "
                              "suggesting possible altered protein-complex formation."})

    if phylo is not None:
        cs = ("strongly supports the functional importance of this sequence" if phylo > 1.5
              else "suggests further functional validation is warranted")
        notes.append({'icon': 'globe', 'label': 'Conservation',
                      'text': f"Cross-species conservation of the AD-specific exon {cs}."})

    if ad_nmd and str(ad_nmd).lower() not in ('false', ''):
        notes.append({'icon': 'triangle-alert', 'label': 'NMD risk',
                      'text': "The AD isoform contains an NMD-sensitive structure, so protein "
                              "translation should be verified by Ribo-seq or mass spectrometry."})

    # ── Headline — one bold synthesis sentence, front-loaded above everything else ──
    def _headline() -> str:
        mech_label = mech_info[0] or mech
        driver = f"a <b>{mech_label}</b> event" if mech_label else "an isoform-usage switch"
        func_bits = []
        if loss_go:
            func_bits.append(f"loses <b>{loss_go[0]['name']}</b> function")
        if gain_go:
            func_bits.append(f"gains <b>{gain_go[0]['name']}</b> function")
        func_str = ' and '.join(func_bits) if func_bits else "shows a usage shift without a single dominant GO change"
        return (f"<b>{conf_label} confidence</b> — this CT→AD case is driven by {driver}; "
                f"the AD-associated isoform {func_str}.")

    # ── Causal pathway steps ──────────────────────────────────────────────────
    pathway = []
    if mech:
        pathway.append(mech_info[0] or mech)
    if all_regs:
        pathway.append(f"TF/ASF activity change ({', '.join(r['gene'] for r in all_regs[:3])})")
    if tss_cls and tss_cls not in ('same_promoter', ''):
        pathway.append(f"Transcription-start-site shift ({tss_cls})")
    if apa_cls and apa_cls not in ('same_apa', ''):
        pathway.append(f"3′-end processing change ({apa_cls})")
    pathway.append("Isoform-ratio switch (DTU)")
    if dg_list or dl_list:
        pathway.append("Domain composition change")
    if gain_go or loss_go:
        pathway.append("GO functional-space reshaping")

    # ── Regulator badge list ──────────────────────────────────────────────────
    reg_out = []
    for r in all_regs[:5]:
        kb = _REGULATOR_KB.get(r.get('gene', ''), ('TF', None, ''))
        reg_out.append({
            'gene': r.get('gene', '?'), 'cat': kb[0] or 'TF', 'known': bool(kb[1]),
            'direction': r.get('direction', ''), 'logFC': r.get('logFC', 0),
            'neg_log10_padj': r.get('neg_log10_padj', 0), 'desc': kb[2],
        })

    tier_key = str(c.get('prism_tier') or '').strip() or _infer_prism_tier(c)
    tier_color, tier_icon, tier_label = _TIER_STYLE.get(
        tier_key, ('#66728A', None, f'Tier · {tier_key or "N/A"}'))

    return {
        'gene': c.get('gene'), 'cell_type': ct_type, 'ct_tx': ct_tx, 'ad_tx': ad_tx,
        'headline': _headline(),
        'tier': {'label': tier_label, 'color': tier_color, 'icon': tier_icon},
        'confidence': {'label': conf_label, 'color': conf_color, 'n': ev_count, 'max': 11},
        'pathway': pathway,
        'regulators': reg_out,
        'context': {
            'tss': tss_cls, 'tss_bp': int(tss_bp) if tss_bp else None,
            'apa': apa_cls, 'apa_bp': int(apa_bp) if apa_bp else None,
            'mech': {'label': mech_info[0] or mech, 'color': mech_info[1], 'detail': mech_info[2]},
        },
        'domains': {
            'gained': [{'name': d, 'func': _domain_func(d)} for d in dg_list],
            'lost':   [{'name': d, 'func': _domain_func(d)} for d in dl_list],
        },
        'af_delta': af_delta,
        'ct_go': ct_top, 'ad_go': ad_top, 'gain_go': gain_go, 'loss_go': loss_go,
        'narrative': notes,
    }
