#!/usr/bin/env python3
"""
Build and persist the FAISS index from a directory of .npy feature vectors,
or append a new entry from a portrait image (auto-detects and aligns face).

Usage (build from .npy features):
    python3 tools/build_faiss_index.py --features data/features \
            --index data/index/faiss.index --labels data/index/labels.json \
            --metric cosine --type flat --gpu 1 --gpu-id 0 --nlist 0 --m 0 --nbits 8

Usage (append from portrait image):
    python3 tools/build_faiss_index.py --add-image /path/to/portrait.jpg \
            --label "Person Name" --engine /path/to/arcface.engine

Defaults are taken from config/config_pipeline.toml if present.
"""
import argparse
import os
import sys
import toml
import cv2
import numpy as np
import configparser
from typing import Optional, Tuple

# Ensure imports work when running this script from any working directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.faiss_index import FaceIndex, FaceIndexConfig
from utils.gen_feature import TensorRTInfer


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
    # New mode: add one portrait image into index
    p.add_argument('--add-image', type=str, default=None, help='Path to a portrait image (not 112x112). Detects, aligns, embeds and appends to index.')
    p.add_argument('--label', type=str, default=None, help='Label/name for the appended entry (default: filename stem)')
    p.add_argument('--engine', type=str, default=None, help='Optional TensorRT ArcFace engine path; if omitted, will resolve from pipeline config')
    return p.parse_args()


def _resolve_engine_path(cfg_path: str) -> str:
    """Resolve ArcFace TensorRT engine path from pipeline TOML -> SGIE config.

    Falls back to common locations under models/arcface.
    """
    engine_path = None
    try:
        cfg_full = cfg_path if os.path.isabs(cfg_path) else os.path.join(REPO_ROOT, cfg_path)
        cfg = toml.load(cfg_full)
        sgie_cfg_path = cfg.get('sgie', {}).get('config-file-path', '').strip()
        if sgie_cfg_path and not os.path.isabs(sgie_cfg_path):
            sgie_cfg_path = os.path.join(os.path.dirname(cfg_full), sgie_cfg_path)
        if sgie_cfg_path and os.path.exists(sgie_cfg_path):
            cp = configparser.ConfigParser()
            cp.read(sgie_cfg_path)
            if cp.has_option('property', 'model-engine-file'):
                engine_path = cp.get('property', 'model-engine-file')
                if engine_path and not os.path.isabs(engine_path):
                    engine_path = os.path.normpath(os.path.join(os.path.dirname(sgie_cfg_path), engine_path))
    except Exception:
        engine_path = None
    # Fallbacks
    if not engine_path or not os.path.exists(engine_path):
        cands = [
            os.path.join(REPO_ROOT, 'models/arcface/glintr100.onnx_b4_gpu0_fp16.engine'),
            os.path.join(REPO_ROOT, 'models/arcface/arcface.engine'),
        ]
        for c in cands:
            if os.path.exists(c):
                engine_path = c
                break
    if not engine_path or not os.path.exists(engine_path):
        raise FileNotFoundError('ArcFace TensorRT engine not found. Provide --engine or configure [sgie] in config_pipeline.toml')
    return engine_path


def _detect_face_bbox_haar(bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Detect the largest frontal face with OpenCV Haar cascade. Returns (x,y,w,h) or None.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Try OpenCV-provided path; fall back to common system path
    cascade_path = None
    try:
        data_attr = getattr(cv2, 'data', None)
        if data_attr is not None:
            hc = getattr(data_attr, 'haarcascades', None)
            if isinstance(hc, str):
                candidate = os.path.join(hc, 'haarcascade_frontalface_default.xml')
                if os.path.exists(candidate):
                    cascade_path = candidate
    except Exception:
        cascade_path = None
    if cascade_path is None or not os.path.exists(cascade_path):
        sys_paths = [
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
        ]
        for p in sys_paths:
            if os.path.exists(p):
                cascade_path = p
                break
    if cascade_path is None or not os.path.exists(cascade_path):
        return None
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60, 60))
    if len(faces) == 0:
        return None
    # choose the largest by area
    x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
    return int(x), int(y), int(w), int(h)


def _crop_align_to_112(bgr: np.ndarray) -> np.ndarray:
    """Crop detected face region (largest) with some padding and resize to 112x112 BGR.
    If no face is detected, center-crop the largest centered square.
    """
    H, W = bgr.shape[:2]
    box = _detect_face_bbox_haar(bgr)
    if box is None:
        # fallback: center square crop
        side = min(H, W)
        cx, cy = W // 2, H // 2
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        crop = bgr[y0:y0+side, x0:x0+side]
    else:
        x, y, w, h = box
        # expand box to include some context and make square
        pad = int(0.2 * max(w, h))
        cx = x + w // 2
        cy = y + h // 2
        side = int(1.2 * max(w, h)) + 2*pad
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        x1 = min(W, x0 + side)
        y1 = min(H, y0 + side)
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            # fallback to original bbox
            crop = bgr[y:y+h, x:x+w]
    if crop.size == 0:
        raise ValueError('Failed to crop face region')
    face112 = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
    return face112


def _arcface_embed_from_bgr(bgr_112: np.ndarray, engine_path: str) -> np.ndarray:
    """Run ArcFace TensorRT engine on a 112x112 BGR image; return L2-normalized 1D embedding.
    """
    # Prepare input: BGR -> RGB, normalize, CHW, NCHW float32 contiguous
    rgb = cv2.cvtColor(bgr_112, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb -= 127.5
    rgb /= 128.0
    chw = np.transpose(rgb, (2, 0, 1))
    inp = np.expand_dims(chw, axis=0).astype(np.float32)
    inp = np.array(inp, dtype=np.float32, order='C')
    # Infer
    trt_model = TensorRTInfer(engine_path, mode='min')
    preds = trt_model.infer(inp)[0]
    emb = np.reshape(preds, (-1,)).astype(np.float32)
    # L2 normalize
    norm = np.linalg.norm(emb) + 1e-12
    emb = emb / norm
    return emb


def add_portrait_image_to_index(image_path: str, label: str, index_path: str, labels_path: str,
                                metric: str = 'cosine', index_type: str = 'flat', use_gpu: bool = True, gpu_id: int = 0,
                                nlist: int = 0, m: int = 0, nbits: int = 8, engine_path: Optional[str] = None):
    """Detect, align and embed a portrait image, then append to FAISS index (creating it if needed)."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    # Resolve engine if not provided
    engine_path = engine_path or _resolve_engine_path('config/config_pipeline.toml')
    # Load and align
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise RuntimeError(f'Failed to read image: {image_path}')
    face112 = _crop_align_to_112(bgr)
    emb = _arcface_embed_from_bgr(face112, engine_path)
    # Prepare index config
    cfg = FaceIndexConfig(
        metric=metric,
        index_type=index_type,
        use_gpu=bool(use_gpu),
        gpu_id=int(gpu_id),
        nlist=(nlist or None),
        m_pq=(m or None),
        nbits_pq=int(nbits),
    )
    # Load or create index
    if os.path.exists(index_path) and os.path.exists(labels_path):
        idx = FaceIndex.load(index_path, labels_path, use_gpu=cfg.use_gpu, gpu_id=cfg.gpu_id)
        # Dim check will happen inside add
    else:
        idx = FaceIndex(dim=len(emb), cfg=cfg)
        idx.build(np.asarray(emb, dtype=np.float32).reshape(1, -1), [label])
        idx.save(index_path, labels_path)
        print(f"Created new index at {index_path} with first entry '{label}'")
        return
    # Append
    idx.add([label], np.asarray(emb, dtype=np.float32).reshape(1, -1))
    idx.save(index_path, labels_path)
    print(f"Appended '{label}' to index. Size now: {idx.size()}")


def main():
    args = parse_args()
    # Resolve paths relative to repository root if not absolute
    features_dir = args.features if os.path.isabs(args.features) else os.path.join(REPO_ROOT, args.features)
    index_path = args.index if os.path.isabs(args.index) else os.path.join(REPO_ROOT, args.index)
    labels_path = args.labels if os.path.isabs(args.labels) else os.path.join(REPO_ROOT, args.labels)

    # Branch: add a portrait image
    if args.add_image:
        label = args.label or os.path.splitext(os.path.basename(args.add_image))[0]
        try:
            # Normalize engine path argument (may be None)
            engine_path_arg: Optional[str] = None
            if args.engine:
                engine_path_arg = args.engine if os.path.isabs(args.engine) else os.path.join(REPO_ROOT, args.engine)
            add_portrait_image_to_index(
                image_path=args.add_image if os.path.isabs(args.add_image) else os.path.join(REPO_ROOT, args.add_image),
                label=label,
                index_path=index_path,
                labels_path=labels_path,
                metric=args.metric,
                index_type=args.index_type,
                use_gpu=bool(args.gpu),
                gpu_id=int(args.gpu_id),
                nlist=int(args.nlist),
                m=int(args.m),
                nbits=int(args.nbits),
                engine_path=engine_path_arg,
            )
        except Exception as e:
            print(f"Failed to add image: {e}", file=sys.stderr)
            sys.exit(1)
        return

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
