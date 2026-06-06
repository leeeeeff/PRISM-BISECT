"""CLI: prism-infer — ESM-2 embeddings → PRISM score matrix (NPY).

This command bridges the gap between upstream long-read assembly and the
PRISM web app.  Two entry modes:

  Mode A — pre-computed ESM-2 embeddings:
    prism-infer --embeddings esm2.npy --ids ids.txt \\
                --model muscle_18go --output scores.npy

  Mode B — protein FASTA (auto-computes ESM-2, requires torch + fair-esm):
    prism-infer --fasta proteins.fasta --ids ids.txt \\
                --model muscle_18go --output scores.npy

Model checkpoints
-----------------
Download from Zenodo (https://doi.org/10.5281/zenodo.PLACEHOLDER):
  muscle_18go/    — 18 BP GO terms, muscle-trained (v15d_bp_clean)
  brain_672go/    — 672 BP GO terms, zero-shot transfer

  wget https://zenodo.org/record/PLACEHOLDER/files/prism_checkpoints.tar.gz
  tar xzf prism_checkpoints.tar.gz -C ~/.prism/

Default checkpoint search path: ~/.prism/  (override with --checkpoint-dir)

Requirements
------------
  pip install tensorflow>=2.13 numpy pandas
  # Only for Mode B (FASTA → ESM-2):
  pip install torch fair-esm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_CHECKPOINT_DIR = Path.home() / '.prism'
ESM2_MODEL_NAME = 'esm2_t30_150M_UR50D'
ESM2_LAYER      = 30

_MODEL_CONFIGS = {
    'muscle_18go': {'n_go': 18, 'tissue': 'muscle'},
    'brain_18go':  {'n_go': 18, 'tissue': 'brain'},
    'brain_672go': {'n_go': 672, 'tissue': 'brain_672'},
}


# ── CLI parsing ───────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='prism-infer',
        description='Run PRISM isoform function inference: embeddings → GO scores.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Input
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument('--embeddings', metavar='NPY',
                     help='Pre-computed ESM-2 embeddings (N × 640 float32 NPY)')
    inp.add_argument('--fasta', metavar='FASTA',
                     help='Protein FASTA — ESM-2 will be computed automatically '
                          '(requires torch + fair-esm)')

    p.add_argument('--ids',      required=True, metavar='FILE',
                   help='Isoform ID list (NPY or TXT, one ID per line)')
    p.add_argument('--model',    default='muscle_18go',
                   choices=list(_MODEL_CONFIGS.keys()),
                   help='PRISM model preset (default: muscle_18go)')
    p.add_argument('--checkpoint-dir', default=str(DEFAULT_CHECKPOINT_DIR),
                   metavar='DIR',
                   help=f'Directory containing model weights '
                        f'(default: {DEFAULT_CHECKPOINT_DIR})')
    p.add_argument('--seeds',    default='0,1,2,3,4',
                   help='Comma-separated seed ensemble to average (default: 0,1,2,3,4)')
    p.add_argument('--batch-size', type=int, default=256,
                   help='Inference batch size (default: 256)')
    p.add_argument('--output',   default='prism_scores.npy', metavar='NPY',
                   help='Output score matrix NPY path (default: prism_scores.npy)')
    p.add_argument('--gene-ids', metavar='FILE',
                   help='Gene ID list (optional, same format as --ids)')
    p.add_argument('--isoform-types', metavar='FILE',
                   help='Isoform type labels (optional, known/nic/nnic)')
    p.add_argument('--device',   default='auto',
                   help='Compute device: auto | cpu | cuda | cuda:N (default: auto)')
    return p


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_ids(path: str):
    import numpy as np
    p = Path(path)
    if p.suffix == '.npy':
        arr = np.load(p, allow_pickle=True)
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])
    return np.array(p.read_text().strip().splitlines())


def _resolve_device(device_arg: str) -> str:
    if device_arg != 'auto':
        return device_arg
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        return 'gpu' if gpus else 'cpu'
    except Exception:
        return 'cpu'


def _check_checkpoint(ckpt_dir: Path, model_name: str, seeds: list[int]) -> list[Path]:
    model_dir = ckpt_dir / model_name
    paths = []
    for s in seeds:
        p = model_dir / f'{model_name}_seed{s}.h5'
        if not p.exists():
            # also try flat naming
            p = model_dir / f'seed{s}.h5'
        if p.exists():
            paths.append(p)
    return paths


# ── ESM-2 embedding (Mode B) ─────────────────────────────────────────────────

def _compute_esm2_embeddings(fasta_path: str, device: str, batch_size: int):
    """Load ESM-2 and compute mean-pool layer-30 embeddings for all sequences."""
    try:
        import torch
        import esm
    except ImportError:
        print(
            '[prism-infer] ERROR: ESM-2 not available.\n'
            '  Install with: pip install fair-esm torch\n'
            '  Or pre-compute embeddings and use --embeddings instead.',
            file=sys.stderr,
        )
        sys.exit(1)

    import numpy as np
    from Bio import SeqIO  # noqa: F401

    print(f'[prism-infer] Loading {ESM2_MODEL_NAME} …')
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    batch_converter  = alphabet.get_batch_converter()
    model.eval()

    dev = torch.device(device if device in ('cpu', 'cuda') else device)
    model = model.to(dev)

    records = list(SeqIO.parse(fasta_path, 'fasta'))
    print(f'[prism-infer] {len(records)} sequences — computing ESM-2 layer {ESM2_LAYER} embeddings …')

    all_embs = []
    for i in range(0, len(records), batch_size):
        batch  = [(r.id, str(r.seq)[:1022]) for r in records[i:i + batch_size]]
        labels, strs, tokens = batch_converter(batch)
        tokens = tokens.to(dev)
        with torch.no_grad():
            out = model(tokens, repr_layers=[ESM2_LAYER], return_contacts=False)
        reps = out['representations'][ESM2_LAYER]  # (B, L+2, 640)
        # mean-pool over sequence positions (exclude BOS/EOS)
        for j, (_, seq) in enumerate(batch):
            L = len(seq)
            emb = reps[j, 1:L + 1].mean(0).cpu().numpy()  # (640,)
            all_embs.append(emb)
        if (i // batch_size) % 10 == 0:
            print(f'[prism-infer]   processed {min(i + batch_size, len(records))}/{len(records)}')

    return np.array(all_embs, dtype=np.float32)


# ── PRISM model architecture ──────────────────────────────────────────────────

def _build_prism_model(n_go: int):
    """Dense → BN → Dropout architecture matching v15d_bp_clean."""
    import tensorflow as tf
    inp = tf.keras.Input(shape=(640,))
    x   = tf.keras.layers.Dense(256, activation='relu')(inp)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Dropout(0.3)(x, training=False)
    x   = tf.keras.layers.Dense(128, activation='relu')(x)
    x   = tf.keras.layers.Dropout(0.2)(x, training=False)
    x   = tf.keras.layers.Dense(64,  activation='relu')(x)
    out = tf.keras.layers.Dense(n_go, activation='sigmoid')(x)
    return tf.keras.Model(inp, out)


# ── Inference ─────────────────────────────────────────────────────────────────

def _run_inference(embeddings, ckpt_paths: list[Path], n_go: int,
                   batch_size: int):
    import numpy as np
    import tensorflow as tf

    if not ckpt_paths:
        print('[prism-infer] ERROR: No checkpoint files found.', file=sys.stderr)
        print('[prism-infer]   Download weights from Zenodo and place in --checkpoint-dir.',
              file=sys.stderr)
        sys.exit(1)

    model = _build_prism_model(n_go)

    all_preds = []
    for ckpt in ckpt_paths:
        print(f'[prism-infer]   Loading checkpoint: {ckpt.name}')
        model.load_weights(str(ckpt))

        preds = []
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            preds.append(model(batch, training=False).numpy())
        all_preds.append(np.vstack(preds))

    return np.mean(all_preds, axis=0).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    import numpy as np

    args   = _build_parser().parse_args(argv)
    seeds  = [int(s) for s in args.seeds.split(',')]
    config = _MODEL_CONFIGS[args.model]
    n_go   = config['n_go']
    ckpt_dir = Path(args.checkpoint_dir)

    print(f'[prism-infer] model={args.model}  n_go={n_go}  seeds={seeds}')

    # ── Load / compute embeddings ─────────────────────────────────────────────
    device = _resolve_device(args.device)
    print(f'[prism-infer] device={device}')

    if args.embeddings:
        print(f'[prism-infer] Loading pre-computed embeddings: {args.embeddings}')
        embeddings = np.load(args.embeddings, allow_pickle=True).astype(np.float32)
    else:
        embeddings = _compute_esm2_embeddings(args.fasta, device, args.batch_size)

    ids = _load_ids(args.ids)
    print(f'[prism-infer] Embeddings: {embeddings.shape}   IDs: {len(ids)}')

    if embeddings.shape[0] != len(ids):
        print(
            f'[prism-infer] ERROR: embedding rows ({embeddings.shape[0]}) '
            f'!= ID count ({len(ids)}). '
            'Ensure --embeddings and --ids correspond to the same isoform list.',
            file=sys.stderr,
        )
        sys.exit(1)

    if embeddings.shape[1] != 640:
        print(
            f'[prism-infer] ERROR: expected 640-dim ESM-2 embeddings, '
            f'got {embeddings.shape[1]}-dim.',
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Find checkpoints ──────────────────────────────────────────────────────
    ckpt_paths = _check_checkpoint(ckpt_dir, args.model, seeds)
    if not ckpt_paths:
        print(
            f'\n[prism-infer] WARNING: No checkpoint files found in '
            f'{ckpt_dir / args.model}/\n'
            f'  Expected:   {ckpt_dir / args.model / f"{args.model}_seed0.h5"}\n\n'
            f'  Download model weights:\n'
            f'    wget https://zenodo.org/record/PLACEHOLDER/files/prism_checkpoints.tar.gz\n'
            f'    tar xzf prism_checkpoints.tar.gz -C {ckpt_dir}\n\n'
            f'  Or set --checkpoint-dir to an existing directory with .h5 files.\n',
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Run inference ─────────────────────────────────────────────────────────
    print(f'[prism-infer] Running inference over {len(ckpt_paths)}-seed ensemble …')
    scores = _run_inference(embeddings, ckpt_paths, n_go, args.batch_size)

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_path = Path(args.output)
    np.save(out_path, scores)
    print(f'[prism-infer] Saved scores: {out_path}  shape={scores.shape}')

    # Save companion files so web app can auto-detect them
    stem = out_path.stem
    base = out_path.parent
    np.save(base / f'{stem}_ids.npy',   np.array(ids,         dtype=str))
    print(f'[prism-infer]   companion IDs → {base / (stem + "_ids.npy")}')

    if args.gene_ids:
        gene_ids = _load_ids(args.gene_ids)
        np.save(base / f'{stem}_gene_ids.npy', np.array(gene_ids, dtype=str))
        print(f'[prism-infer]   gene IDs → {base / (stem + "_gene_ids.npy")}')

    if args.isoform_types:
        iso_types = _load_ids(args.isoform_types)
        np.save(base / f'{stem}_types.npy', np.array(iso_types, dtype=str))
        print(f'[prism-infer]   isoform types → {base / (stem + "_types.npy")}')

    # ── Metadata JSON ─────────────────────────────────────────────────────────
    meta = {
        'model':      args.model,
        'n_isoforms': int(embeddings.shape[0]),
        'n_go':       n_go,
        'seeds':      seeds,
        'tissue':     config['tissue'],
        'input_mode': 'embeddings' if args.embeddings else 'fasta',
        'checkpoints': [str(p) for p in ckpt_paths],
    }
    meta_path = base / f'{stem}_meta.json'
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f'[prism-infer]   metadata → {meta_path}')
    print('\n[prism-infer] Done. Upload these files to the PRISM web app:')
    print(f'  Score matrix : {out_path}')
    print(f'  Isoform IDs  : {base / (stem + "_ids.npy")}')


if __name__ == '__main__':
    main()
