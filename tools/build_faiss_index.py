#!/usr/bin/env python3
"""
Build and persist the FAISS index from a directory of .npy feature vectors.

Usage:
  python3 tools/build_faiss_index.py --features data/features \
      --index data/index/faiss.index --labels data/index/labels.json \
      --metric cosine --type flat --gpu 1 --gpu-id 0 --nlist 0 --m 0 --nbits 8

Defaults are taken from config/config_pipeline.toml if present.
"""
import argparse
import os
import sys
import toml

# Ensure imports work when running this script from any working directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.faiss_index import FaceIndex, FaceIndexConfig


def load_cfg_defaults(cfg_path: str):
    try:
        # Resolve config path relative to repo root if not absolute
        cfg_full = cfg_path if os.path.isabs(cfg_path) else os.path.join(REPO_ROOT, cfg_path)
        cfg = toml.load(cfg_full)
        recog = cfg.get('recognition', {})
        return {
            'features': str(recog.get('feature_dir', 'data/features')),
            'index': str(recog.get('index_path', 'data/index/faiss.index')),
            'labels': str(recog.get('labels_path', 'data/index/labels.json')),
            'metric': str(recog.get('metric', 'cosine')),
            'type': str(recog.get('index_type', 'flat')),
            'gpu': int(recog.get('use_gpu', 1)),
            'gpu_id': int(recog.get('gpu_id', 0)),
            'nlist': int(recog.get('nlist', 0)),
            'm': int(recog.get('m_pq', 0)),
            'nbits': int(recog.get('nbits_pq', 8)),
        }
    except Exception:
        return {
            'features': 'data/features',
            'index': 'data/index/faiss.index',
            'labels': 'data/index/labels.json',
            'metric': 'cosine',
            'type': 'flat',
            'gpu': 1,
            'gpu_id': 0,
            'nlist': 0,
            'm': 0,
            'nbits': 8,
        }


def parse_args():
    defaults = load_cfg_defaults('config/config_pipeline.toml')
    p = argparse.ArgumentParser(description='Build FAISS index from feature .npy files')
    p.add_argument('--features', default=defaults['features'], help='Directory with .npy feature files')
    p.add_argument('--index', default=defaults['index'], help='Output index path')
    p.add_argument('--labels', default=defaults['labels'], help='Output labels json path')
    p.add_argument('--metric', default=defaults['metric'], choices=['cosine', 'l2'])
    p.add_argument('--type', dest='index_type', default=defaults['type'], choices=['flat', 'ivf', 'ivfpq'])
    p.add_argument('--gpu', type=int, default=defaults['gpu'])
    p.add_argument('--gpu-id', type=int, default=defaults['gpu_id'])
    p.add_argument('--nlist', type=int, default=defaults['nlist'])
    p.add_argument('--m', type=int, default=defaults['m'])
    p.add_argument('--nbits', type=int, default=defaults['nbits'])
    p.add_argument('--dry-run', action='store_true', help='Print resolved paths and config then exit')
    return p.parse_args()


def main():
    args = parse_args()
    # Resolve paths relative to repository root if not absolute
    features_dir = args.features if os.path.isabs(args.features) else os.path.join(REPO_ROOT, args.features)
    index_path = args.index if os.path.isabs(args.index) else os.path.join(REPO_ROOT, args.index)
    labels_path = args.labels if os.path.isabs(args.labels) else os.path.join(REPO_ROOT, args.labels)

    if not os.path.isdir(features_dir):
        print(f"Features directory not found: {features_dir}", file=sys.stderr)
        sys.exit(1)

    cfg = FaceIndexConfig(
        metric=args.metric,
        index_type=args.index_type,
        use_gpu=bool(args.gpu),
        gpu_id=int(args.gpu_id),
        nlist=(args.nlist or None),
        m_pq=(args.m or None),
        nbits_pq=int(args.nbits),
    )

    print(f"Building index from {features_dir} (metric={args.metric}, type={args.index_type}, gpu={args.gpu})")

    if args.dry_run:
        print("Dry run: configuration validated; exiting without building index.")
        return

    idx = FaceIndex.from_dir(features_dir, cfg)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    idx.save(index_path, labels_path)
    print(f"Saved index to {index_path}")
    print(f"Saved labels to {labels_path}")
    print(f"Vectors indexed: {idx.size()}")


if __name__ == '__main__':
    main()
