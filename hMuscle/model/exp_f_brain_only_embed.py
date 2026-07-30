#!/usr/bin/env python3
"""Brain-only embedding script: reuse trained 650M model to embed 36748 BambuTx isoforms."""
import os, re, time, argparse
import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
BRAIN_PEP = os.path.join(DATA_DIR, 'top30k_isoforms.pep')
BRAIN_IDS = 'my_isoform_list_fixed.npy'
MAX_LEN   = 1022

TYPE_RANK = {'complete': 4, '5prime_partial': 3, '3prime_partial': 2, 'internal': 1}

def clean_id(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

def load_brain_seqs(pep_path, brain_ids, max_len=MAX_LEN):
    id_set  = set(brain_ids)
    records = {}
    cur_id = cur_meta = None; cur_seq = []
    def flush():
        nonlocal cur_id, cur_meta, cur_seq
        if cur_id is None or cur_id not in id_set: return
        seq = ''.join(cur_seq).replace('*','').strip()
        if not seq: return
        rank, score, length = cur_meta
        prev = records.get(cur_id)
        if prev is None or (rank,score,length) > prev[:3]:
            records[cur_id] = (rank, score, length, seq[:max_len])
    with open(pep_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush(); cur_seq = []
                m_id = re.match(r'>(\S+)', line)
                if not m_id: cur_id = None; continue
                cur_id = re.sub(r'\.p\d+$', '', m_id.group(1))
                m_type = re.search(r'ORF type:(\S+)', line)
                m_score = re.search(r'score=([\d.]+)', line)
                m_len   = re.search(r'len:(\d+)', line)
                orf_t   = (m_type.group(1) if m_type else 'internal').split('(')[0]
                cur_meta = (TYPE_RANK.get(orf_t,1),
                            float(m_score.group(1)) if m_score else 0.0,
                            int(m_len.group(1)) if m_len else 0)
            else:
                cur_seq.append(line.strip())
    flush()
    return {k: v[3] for k, v in records.items()}

def embed_esm2_brain(model_name, tag, n_layers, layers, batch_size, device):
    import esm
    loader = getattr(esm.pretrained, model_name)
    model, alphabet = loader()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded {model_name}  params={n_params:.0f}M  layers={n_layers}")

    brain_ids = [clean_id(x) for x in np.load(BRAIN_IDS, allow_pickle=True)]
    seq_dict  = load_brain_seqs(BRAIN_PEP, brain_ids)
    N = len(brain_ids)
    print(f"  {len(seq_dict)}/{N} sequences found")

    valid = [(i, iid, seq_dict[iid]) for i, iid in enumerate(brain_ids) if iid in seq_dict]
    n_miss = N - len(valid)
    if n_miss: print(f"  [WARN] {n_miss}/{N} missing → zero vectors")

    mats = None
    n_batches = (len(valid) + batch_size - 1) // batch_size
    t0 = time.time()
    print(f"  Extracting L{layers} from {len(valid)} seqs in {n_batches} batches...", flush=True)

    for b in range(n_batches):
        batch   = valid[b*batch_size:(b+1)*batch_size]
        indices = [x[0] for x in batch]
        seqs    = [(x[1], x[2]) for x in batch]
        _, _, tokens = batch_converter(seqs)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=layers, return_contacts=False)
        if mats is None:
            dim  = out['representations'][layers[0]].shape[-1]
            mats = {L: np.zeros((N, dim), dtype=np.float32) for L in layers}
        for L in layers:
            reps = out['representations'][L]
            for k, (_, seq) in enumerate(seqs):
                mats[L][indices[k]] = reps[k, 1:len(seq)+1, :].float().mean(0).cpu().numpy()
        if (b+1) % 20 == 0 or b == n_batches-1:
            el = time.time()-t0; eta = el/(b+1)*(n_batches-b-1)
            gpu = torch.cuda.memory_allocated(device)/1e9 if torch.cuda.is_available() else 0
            print(f"  [{b+1:4d}/{n_batches}] {el:.0f}s ETA={eta:.0f}s GPU={gpu:.1f}GB", flush=True)

    for L in layers:
        out_path = f'{DATA_DIR}/esm2_layer_{L:02d}_{tag}.npy'
        np.save(out_path, mats[L])
        norms = np.linalg.norm(mats[L], axis=1)
        print(f"  Saved {out_path}  shape={mats[L].shape}  norm μ={norms.mean():.3f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='650M', choices=['650M', '3B'])
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--gpu',   type=int, default=1)
    args = p.parse_args()

    cfg = {
        '650M': ('esm2_t33_650M_UR50D', 't33_650M', 33, [16, 33]),
        '3B':   ('esm2_t36_3B_UR50D',   't36_3B',   36, [18, 36]),
    }[args.model]
    model_name, tag, n_layers, layers = cfg
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"[brain-only] {args.model}  device={device}  batch={args.batch}")
    embed_esm2_brain(model_name, tag, n_layers, layers, args.batch, device)
    print("[DONE]")

if __name__ == '__main__':
    main()
