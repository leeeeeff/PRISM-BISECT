#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_f_plm_scale_embed.py
------------------------
PLM 범용성 실험 — 임베딩 계산 단계.

지원 모델:
  ESM-2 가족 (fair-esm 2.0.0):
    650M   esm2_t33_650M_UR50D   33L  1280-dim
    3B     esm2_t36_3B_UR50D     36L  2560-dim
    (150M 기준값은 이미 존재)

  ProtTrans 가족 (transformers):
    ProtT5  Rostlab/prot_t5_xl_uniref50  24L  1024-dim

  ESM3 (EvolutionaryScale esm package, 선택적):
    ESM3    EvolutionaryScale/esm3-sm-open-v1  48L  1536-dim

각 모델에서 두 레이어만 추출:
  L_final = 마지막 레이어
  L_mid   = L_final // 2  (50% depth)

실행 예시:
  cd hMuscle/model/
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate isoform_env

  # GPU 0: 650M (~40min)
  CUDA_VISIBLE_DEVICES=0 nohup python3 -u exp_f_plm_scale_embed.py --model 650M \\
    > ../../logs_isoform/exp_f_embed_650M_$(date +%Y%m%d_%H%M).log 2>&1 &

  # GPU 0: 3B (~3h, after 650M)
  CUDA_VISIBLE_DEVICES=0 nohup python3 -u exp_f_plm_scale_embed.py --model 3B --batch 8 \\
    > ../../logs_isoform/exp_f_embed_3B_$(date +%Y%m%d_%H%M).log 2>&1 &

  # GPU 1: ProtT5 (~50min)
  CUDA_VISIBLE_DEVICES=1 nohup python3 -u exp_f_plm_scale_embed.py --model ProtT5 \\
    > ../../logs_isoform/exp_f_embed_ProtT5_$(date +%Y%m%d_%H%M).log 2>&1 &

  # GPU 1: ESM3 (~1h, requires: pip install esm)
  CUDA_VISIBLE_DEVICES=1 nohup python3 -u exp_f_plm_scale_embed.py --model ESM3 \\
    > ../../logs_isoform/exp_f_embed_ESM3_$(date +%Y%m%d_%H%M).log 2>&1 &
"""

import os, re, sys, time, argparse
import numpy as np
import torch
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
TRAIN_SEQ = os.path.join(DATA_DIR,
            'raw_data/data/raw_data/sequence_data/isoform_cds_sequences.txt')
TRAIN_ISO = os.path.join(DATA_DIR,
            'raw_data/data/id_lists/train_isoform_list.npy')
BRAIN_PEP = os.path.join(DATA_DIR, 'top30k_isoforms.pep')  # BambuTx protein sequences
BRAIN_IDS = 'my_isoform_list_fixed.npy'

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_CFG = {
    '650M': {
        'family': 'esm2',
        'esm_name': 'esm2_t33_650M_UR50D',
        'n_layers': 33, 'dim': 1280,
        'tag': 't33_650M',
    },
    '3B': {
        'family': 'esm2',
        'esm_name': 'esm2_t36_3B_UR50D',
        'n_layers': 36, 'dim': 2560,
        'tag': 't36_3B',
    },
    'ProtT5': {
        'family': 'prottrans',
        'hf_name': 'Rostlab/prot_t5_xl_uniref50',
        'n_layers': 24, 'dim': 1024,
        'tag': 'prot_t5_xl',
    },
    'ESM3': {
        'family': 'esm3',
        'hf_name': 'EvolutionaryScale/esm3-sm-open-v1',
        'n_layers': 48, 'dim': 1536,
        'tag': 'esm3_sm',
    },
    'Ankh': {
        'family': 'ankh',
        'hf_name': 'ElnaggarLab/ankh-base',
        'n_layers': 48, 'dim': 768,
        'tag': 'ankh_base',
    },
}

MAX_LEN    = 1022
BATCH_SIZE = 32


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',  choices=list(MODEL_CFG.keys()), default='650M')
    p.add_argument('--gpu',    type=int, default=0)
    p.add_argument('--batch',  type=int, default=BATCH_SIZE)
    p.add_argument('--fp16',   action='store_true')
    return p.parse_args()


# ── Sequence parsers ───────────────────────────────────────────────────────────
def parse_cds(path, max_len=MAX_LEN):
    """Header format: >GENE_SYMBOL|NM_ACCESSION — use second field as key."""
    records = {}
    cur_key, cur_seq = None, []
    def flush():
        if cur_key and cur_seq:
            seq = ''.join(cur_seq).replace('*', '').strip()
            if seq:
                records[cur_key] = seq[:max_len]
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush(); cur_seq = []
                m = re.match(r'>(\S+)\|(\S+)', line)
                cur_key = m.group(2) if m else None
            else:
                cur_seq.append(line)
    flush()
    return records


def clean_id(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s


def load_brain_seqs(pep_path, brain_ids, max_len=MAX_LEN):
    """Load BambuTx protein sequences from top30k_isoforms.pep TransDecoder file.
    Header: >BambuTx10.p1 GENE.BambuTx10~~BambuTx10.p1  ORF type:complete ...
    Key: BambuTx10 (strip .p{N} suffix, keep highest-rank ORF).
    """
    id_set  = set(brain_ids)
    records = {}  # {bambu_id: (rank, score, length, seq)}
    TYPE_RANK = {'complete': 4, '5prime_partial': 3, '3prime_partial': 2, 'internal': 1}
    cur_id = cur_meta = None
    cur_seq = []

    def flush():
        nonlocal cur_id, cur_meta, cur_seq
        if cur_id is None or cur_id not in id_set:
            return
        seq = ''.join(cur_seq).replace('*', '').strip()
        if not seq:
            return
        rank, score, length = cur_meta
        prev = records.get(cur_id)
        if prev is None or (rank, score, length) > prev[:3]:
            records[cur_id] = (rank, score, length, seq[:max_len])

    try:
        with open(pep_path) as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('>'):
                    flush(); cur_seq = []
                    m_id    = re.match(r'>(\S+)', line)
                    if not m_id:
                        cur_id = None; continue
                    raw_id  = m_id.group(1)  # BambuTx10.p1
                    cur_id  = re.sub(r'\.p\d+$', '', raw_id)  # → BambuTx10
                    m_type  = re.search(r'ORF type:(\S+)', line)
                    m_score = re.search(r'score=([\d.]+)', line)
                    m_len   = re.search(r'len:(\d+)', line)
                    orf_t   = (m_type.group(1) if m_type else 'internal').split('(')[0]
                    cur_meta = (TYPE_RANK.get(orf_t, 1),
                                float(m_score.group(1)) if m_score else 0.0,
                                int(m_len.group(1)) if m_len else 0)
                else:
                    cur_seq.append(line.strip())
        flush()
    except Exception as e:
        print(f"  [WARN] Brain PEP: {e}")

    return {k: v[3] for k, v in records.items()}


# ── ESM-2 embedding ───────────────────────────────────────────────────────────
@torch.no_grad()
def embed_esm2_batch(model, batch_converter, sequences, device, layers, fp16=False):
    _, _, tokens = batch_converter(sequences)
    tokens = tokens.to(device)
    ctx = torch.cuda.amp.autocast(enabled=fp16)
    with ctx:
        out = model(tokens, repr_layers=layers, return_contacts=False)
    result = {}
    for L in layers:
        reps = out['representations'][L]
        embs = []
        for i, (_, seq) in enumerate(sequences):
            embs.append(reps[i, 1:len(seq)+1, :].float().mean(0).cpu().numpy())
        result[L] = np.array(embs, dtype=np.float32)
    return result


def load_esm2_model(cfg, device, fp16=False):
    import esm
    loader = getattr(esm.pretrained, cfg['esm_name'])
    model, alphabet = loader()
    if fp16:
        model = model.half()
    model = model.eval().to(device)
    return model, alphabet.get_batch_converter()


# ── ProtT5 embedding ──────────────────────────────────────────────────────────
@torch.no_grad()
def embed_prottrans_batch(model, tokenizer, sequences, device, layers, fp16=False):
    """
    sequences: list of (id, aa_seq) tuples.
    ProtT5 requires spaces between amino acids and uppercase letters only.
    Unknown amino acids (B, J, O, U, Z) replaced by X.
    """
    UNKNOWN = re.compile(r'[BJOUZ]')
    seqs_clean  = [UNKNOWN.sub('X', s.upper()) for _, s in sequences]
    seqs_spaced = [' '.join(list(s)) for s in seqs_clean]

    encoded = tokenizer(
        seqs_spaced,
        add_special_tokens=True,
        padding='longest',
        return_tensors='pt',
        truncation=True,
        max_length=MAX_LEN + 2,
    )
    input_ids      = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.cuda.amp.autocast(enabled=fp16):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    # hidden_states: tuple of (n_layers+1) tensors [batch, seq_len, 1024]
    # index 0 = embedding layer, 1..24 = transformer layers
    result = {}
    for L in layers:
        hs   = out.hidden_states[L].float()  # [B, T, D]
        embs = []
        for i, mask_row in enumerate(attention_mask):
            # exclude padding and special tokens (last token is </s>)
            valid = mask_row.bool()
            valid[-1] = False  # exclude EOS
            embs.append(hs[i][valid].mean(0).cpu().numpy())
        result[L] = np.array(embs, dtype=np.float32)
    return result


def load_prottrans_model(cfg, device, fp16=False):
    from transformers import T5EncoderModel, T5Tokenizer
    print(f"  Loading {cfg['hf_name']} ...")
    tokenizer = T5Tokenizer.from_pretrained(cfg['hf_name'], do_lower_case=False)
    model     = T5EncoderModel.from_pretrained(cfg['hf_name'])
    if fp16:
        model = model.half()
    model = model.eval().to(device)
    return model, tokenizer


# ── Ankh embedding ────────────────────────────────────────────────────────────
_STD_AA = set('ACDEFGHIKLMNPQRSTVWY')

@torch.no_grad()
def embed_ankh_batch(model, tokenizer, sequences, device, layers, fp16=False):
    """Ankh (T5 encoder): tokenizer expects sequences pre-split into AA tokens
    (is_split_into_words=True). Mean-pool over valid tokens, excluding pad + EOS."""
    seqs_list = [[c if c in _STD_AA else 'X' for c in s.upper()]
                 for _, s in sequences]
    enc = tokenizer.batch_encode_plus(
        seqs_list, add_special_tokens=True, padding=True,
        is_split_into_words=True, return_tensors='pt',
        truncation=True, max_length=MAX_LEN + 1)
    input_ids = enc['input_ids'].to(device)
    attn_mask = enc['attention_mask'].to(device)
    eos_id = tokenizer.eos_token_id
    with torch.cuda.amp.autocast(enabled=fp16):
        out = model(input_ids=input_ids, attention_mask=attn_mask,
                    output_hidden_states=True)
    result = {}
    for L in layers:
        hs = out.hidden_states[L].float()  # [B, T, D]
        embs = []
        for i in range(hs.shape[0]):
            valid = attn_mask[i].bool() & (input_ids[i] != eos_id)
            embs.append(hs[i][valid].mean(0).cpu().numpy())
        result[L] = np.array(embs, dtype=np.float32)
    return result


def load_ankh_model(cfg, device, fp16=False):
    from transformers import T5EncoderModel, AutoTokenizer
    print(f"  Loading {cfg['hf_name']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg['hf_name'])
    model     = T5EncoderModel.from_pretrained(cfg['hf_name'])
    if fp16:
        model = model.half()
    model = model.eval().to(device)
    return model, tokenizer


# ── ESM3 embedding ────────────────────────────────────────────────────────────
def load_esm3_model(cfg, device, fp16=False):
    """
    ESM3 open model via EvolutionaryScale esm package.
    Install: pip install esm   (NOT fair-esm)
    Note: may conflict with fair-esm; use a separate env if needed.
    """
    try:
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein
        print(f"  Loading ESM3 from EvolutionaryScale package...")
        model = ESM3.from_pretrained('esm3_sm_open_v1')
        model = model.eval().to(device)
        return model, None  # no separate tokenizer
    except ImportError:
        try:
            from transformers import AutoTokenizer, AutoModel
            print(f"  Loading ESM3 from HuggingFace: {cfg['hf_name']} ...")
            tokenizer = AutoTokenizer.from_pretrained(cfg['hf_name'])
            model     = AutoModel.from_pretrained(cfg['hf_name'],
                                                  output_hidden_states=True)
            if fp16:
                model = model.half()
            model = model.eval().to(device)
            return model, tokenizer
        except Exception as e:
            print(f"  [ERROR] ESM3 not available: {e}")
            print("  Install via: pip install esm")
            print("  OR: pip install transformers (for HuggingFace version)")
            sys.exit(1)


@torch.no_grad()
def embed_esm3_batch(model, tokenizer, sequences, device, layers, fp16=False):
    if tokenizer is None:
        # EvolutionaryScale API
        from esm.sdk.api import ESMProtein
        result = {L: [] for L in layers}
        for _, seq in sequences:
            protein = ESMProtein(sequence=seq)
            out     = model.encode(protein, output_hidden_states=True)
            hs      = out.sequence_hidden_states  # (n_layers, seq_len, dim)
            for L in layers:
                result[L].append(hs[L - 1].mean(0).cpu().float().numpy())
        return {L: np.array(result[L], dtype=np.float32) for L in layers}
    else:
        # HuggingFace API
        seqs_text  = [seq for _, seq in sequences]
        encoded    = tokenizer(seqs_text, return_tensors='pt', padding=True,
                               truncation=True, max_length=MAX_LEN + 2)
        input_ids  = encoded['input_ids'].to(device)
        attn_mask  = encoded['attention_mask'].to(device)
        with torch.cuda.amp.autocast(enabled=fp16):
            out = model(input_ids=input_ids, attention_mask=attn_mask,
                        output_hidden_states=True)
        result = {}
        for L in layers:
            hs   = out.hidden_states[L].float()
            embs = []
            for i, mask_row in enumerate(attn_mask):
                valid = mask_row.bool()
                valid[-1] = False
                embs.append(hs[i][valid].mean(0).cpu().numpy())
            result[L] = np.array(embs, dtype=np.float32)
        return result


# ── Generic compute-and-save ───────────────────────────────────────────────────
def compute_and_save(embed_fn, model, aux, iso_ids, seq_dict, device,
                     layers_to_extract, out_paths, batch_size, fp16=False):
    N    = len(iso_ids)
    skip = [L for L in layers_to_extract if os.path.exists(out_paths[L])]
    todo = [L for L in layers_to_extract if not os.path.exists(out_paths[L])]
    if skip:
        print(f"  [SKIP] Already exists: L{skip}")
    if not todo:
        return

    # Probe dim
    dim = None

    valid = [(i, iid, seq_dict[iid]) for i, iid in enumerate(iso_ids) if iid in seq_dict]
    n_miss = N - len(valid)
    if n_miss > 0:
        print(f"  [WARN] {n_miss}/{N} isoforms missing sequence → zero vectors")

    mats      = None  # allocated after first batch
    n_batches = (len(valid) + batch_size - 1) // batch_size
    t0 = time.time()
    print(f"  Extracting L{todo} from {len(valid)} seqs in {n_batches} batches...", flush=True)

    for b in range(n_batches):
        batch   = valid[b * batch_size: (b + 1) * batch_size]
        indices = [x[0] for x in batch]
        seqs    = [(x[1], x[2]) for x in batch]

        try:
            emb_dict = embed_fn(model, aux, seqs, device, todo, fp16)
        except RuntimeError as e:
            if 'memory' in str(e).lower():
                torch.cuda.empty_cache()
                half = max(1, batch_size // 2)
                emb_dict = {}
                for s in range(0, len(batch), half):
                    sub      = batch[s: s + half]
                    sub_seqs = [(x[1], x[2]) for x in sub]
                    sub_emb  = embed_fn(model, aux, sub_seqs, device, todo, fp16)
                    for L in todo:
                        if L not in emb_dict:
                            emb_dict[L] = np.zeros((len(batch), sub_emb[L].shape[1]),
                                                   dtype=np.float32)
                        emb_dict[L][s: s + len(sub)] = sub_emb[L]
            else:
                raise

        if mats is None:
            dim  = emb_dict[todo[0]].shape[1]
            mats = {L: np.zeros((N, dim), dtype=np.float32) for L in todo}

        for L in todo:
            for k, idx in enumerate(indices):
                mats[L][idx] = emb_dict[L][k]

        if (b + 1) % 20 == 0 or b == n_batches - 1:
            el  = time.time() - t0
            eta = el / (b + 1) * (n_batches - b - 1)
            gpu = (torch.cuda.memory_allocated(device) / 1e9
                   if torch.cuda.is_available() else 0)
            print(f"  [{b+1:4d}/{n_batches}] {el:.0f}s ETA={eta:.0f}s GPU={gpu:.1f}GB",
                  flush=True)

    for L in todo:
        np.save(out_paths[L], mats[L])
        norms = np.linalg.norm(mats[L], axis=1)
        print(f"  Saved {out_paths[L]}  shape={mats[L].shape}  "
              f"norm μ={norms.mean():.3f} σ={norms.std():.3f}", flush=True)


# ── Output path helpers ────────────────────────────────────────────────────────
def get_out_paths(tag, layers, split='train'):
    paths = {}
    for L in layers:
        if split == 'train':
            paths[L] = f'{DATA_DIR}/esm2_train_human_layer{L:02d}_{tag}.npy'
        else:
            paths[L] = f'{DATA_DIR}/esm2_layer_{L:02d}_{tag}.npy'
    return paths


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    cfg    = MODEL_CFG[args.model]
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    family = cfg['family']

    L_final = cfg['n_layers']
    L_mid   = L_final // 2
    layers  = sorted({L_mid, L_final})
    tag     = cfg['tag']

    print("=" * 70)
    print(f"  PLM Generalization Embed: {args.model}  ({cfg.get('esm_name', cfg.get('hf_name'))})")
    print(f"  Layers: L{L_final} (final) + L{L_mid} (mid={L_mid}/{L_final})")
    print(f"  device={device}  batch={args.batch}  fp16={args.fp16}")
    print("=" * 70, flush=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[1] Loading model...")
    if family == 'esm2':
        model, aux = load_esm2_model(cfg, device, args.fp16)
        embed_fn   = embed_esm2_batch
    elif family == 'prottrans':
        model, aux = load_prottrans_model(cfg, device, args.fp16)
        embed_fn   = embed_prottrans_batch
    elif family == 'esm3':
        model, aux = load_esm3_model(cfg, device, args.fp16)
        embed_fn   = embed_esm3_batch
    elif family == 'ankh':
        model, aux = load_ankh_model(cfg, device, args.fp16)
        embed_fn   = embed_ankh_batch
    else:
        raise ValueError(f"Unknown family: {family}")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded  params={n_params:.0f}M  dim={cfg['dim']}  layers={cfg['n_layers']}")

    # ── Muscle train set ──────────────────────────────────────────────────────
    print("\n[2] Muscle train set...")
    train_seq  = parse_cds(TRAIN_SEQ)
    train_ids  = [clean_id(x) for x in np.load(TRAIN_ISO, allow_pickle=True)]
    print(f"  {len(train_seq)} CDS seqs, {len(train_ids)} train isoforms")

    compute_and_save(
        embed_fn, model, aux, train_ids, train_seq, device,
        layers, get_out_paths(tag, layers, 'train'), args.batch, args.fp16,
    )

    # ── Brain test set (zero-shot) ────────────────────────────────────────────
    print("\n[3] Brain test set (zero-shot)...")
    brain_ids = [clean_id(x) for x in np.load(BRAIN_IDS, allow_pickle=True)]
    brain_seq = load_brain_seqs(BRAIN_PEP, brain_ids)
    print(f"  {len(brain_seq)}/{len(brain_ids)} seqs found")

    if len(brain_seq) == 0:
        print("  [SKIP] No brain sequences found — skipping brain embedding.")
        print("  (SQANTI3 CSV uses ENST IDs; BambuTx IDs require separate pipeline)")
    else:
        compute_and_save(
            embed_fn, model, aux, brain_ids, brain_seq, device,
            layers, get_out_paths(tag, layers, 'test'), args.batch, args.fp16,
        )

    print(f"\n[DONE] {args.model} embeddings saved.", flush=True)


if __name__ == '__main__':
    main()
