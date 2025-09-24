import os
import time
import json
import numpy as np
from typing import Optional, Tuple, Dict, Any

from utils.faiss_index import FaceIndex, FaceIndexConfig
from utils.parser_cfg import safe_load_index

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # noqa

# ---------------- Configuration Helpers ---------------- #

def resolve_engine_from_cfg(cfg: dict) -> Optional[str]:
    try:
        sgie_path = cfg.get('sgie', {}).get('config-file-path', '') if isinstance(cfg, dict) else ''
        if sgie_path and not os.path.isabs(sgie_path):
            sgie_path = os.path.join(os.getcwd(), sgie_path)
        if sgie_path and os.path.exists(sgie_path):
            import configparser
            cp = configparser.ConfigParser()
            cp.read(sgie_path)
            if cp.has_option('property', 'model-engine-file'):
                eng = cp.get('property', 'model-engine-file')
                if eng and not os.path.isabs(eng):
                    eng = os.path.join(os.path.dirname(sgie_path), eng)
                if os.path.exists(eng):
                    return eng
    except Exception:
        pass
    # fallback search
    candidates = [
        'models/arcface/glintr100.onnx_b4_gpu0_fp16.engine',
        'models/arcface/arcface.engine'
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None

# ---------------- Face Alignment & Embedding ---------------- #

def haar_face_bbox(bgr):
    if cv2 is None:
        return None
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cascade_paths = []
        try:
            data_attr = getattr(cv2, 'data', None)
            if data_attr is not None:
                hc = getattr(data_attr, 'haarcascades', None)
                if isinstance(hc, str):
                    cascade_paths.append(os.path.join(hc, 'haarcascade_frontalface_default.xml'))
        except Exception:
            pass
        cascade_paths += [
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
        ]
        cascade = None
        for p in cascade_paths:
            if os.path.exists(p):
                cascade = cv2.CascadeClassifier(p)
                break
        if cascade is None:
            return None
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60,60))
        if len(faces) == 0:
            return None
        return max(faces, key=lambda r: r[2]*r[3])
    except Exception:
        return None

def crop_align_112(bgr, margin: float=0.10):
    if bgr is None or bgr.size == 0:
        return None
    H, W = bgr.shape[:2]
    box = haar_face_bbox(bgr)
    if box is None:
        # center square fallback
        side = min(H, W)
        cx, cy = W//2, H//2
        x0 = max(0, cx - side//2); y0 = max(0, cy - side//2)
        crop = bgr[y0:y0+side, x0:x0+side]
    else:
        # Use detected face bounding box EXACTLY as detected
        # This preserves the mouth region and matches the good detection shown in crop analysis
        x, y, w, h = box
        
        # Ensure the crop rect is within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, W - x)
        h = min(h, H - y)
        
        if w > 0 and h > 0:
            crop = bgr[y:y+h, x:x+w]
        else:
            crop = None
    
    if crop is None or crop.size == 0:
        return None
    try:
        face112 = cv2.resize(crop, (112,112), interpolation=cv2.INTER_LINEAR)
        return face112
    except Exception:
        return None

def embed_arcface(face112_bgr, engine_path: str):
    from utils.gen_feature import TensorRTInfer
    if cv2 is None:
        raise RuntimeError('cv2 not available for embedding prep')
    rgb = cv2.cvtColor(face112_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb -= 127.5; rgb /= 128.0
    chw = np.transpose(rgb, (2,0,1))
    inp = np.expand_dims(chw,0).astype(np.float32)
    model = TensorRTInfer(engine_path, mode='min')
    out = model.infer(np.array(inp, dtype=np.float32, order='C'))[0]
    emb = np.asarray(out).reshape(-1).astype(np.float32)
    n = float(np.linalg.norm(emb))+1e-12
    return (emb/n).astype(np.float32)

# ---------------- Index & Metadata Ops ---------------- #

def load_or_create_index(recog_cfg: dict) -> FaceIndex:
    idx = safe_load_index(recog_cfg)
    if idx is not None:
        return idx
    # create empty
    try:
        metric = recog_cfg.get('metric','cosine')
        index_type = recog_cfg.get('index_type','flat')
        use_gpu = bool(int(recog_cfg.get('use_gpu',0)))
        gpu_id = int(recog_cfg.get('gpu_id',0))
    except Exception:
        metric, index_type, use_gpu, gpu_id = 'cosine','flat',False,0
    cfg_idx = FaceIndexConfig(metric=metric, index_type=index_type, use_gpu=use_gpu, gpu_id=gpu_id)
    idx = FaceIndex(dim=512, cfg=cfg_idx)
    cpu_index, _ = idx._make_faiss_index(0)
    idx._cpu_index = cpu_index
    idx._gpu_index = idx._to_gpu(cpu_index)
    idx._labels = []
    return idx

def read_labels(labels_path: str) -> Dict[str, Any]:
    if not labels_path or not os.path.exists(labels_path):
        return {}
    try:
        with open(labels_path,'r',encoding='utf-8') as f:
            return json.load(f) or {}
    except Exception:
        return {}

def write_labels(labels_path: str, meta: Dict[str, Any]):
    if not labels_path:
        return
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    with open(labels_path,'w',encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def upsert_person_meta(meta: Dict[str, Any], user_id: str, name: str, aligned_rel: Optional[str]):
    persons = meta.get('persons', {}) if isinstance(meta.get('persons', {}), dict) else {}
    now_ts = int(time.time())
    rec = persons.get(user_id, {}) if isinstance(persons.get(user_id, {}), dict) else {}
    # User preference: initialize times to 0 if absent
    if 'start_time' not in rec:
        rec['start_time'] = 0
    if 'end_time' not in rec:
        rec['end_time'] = 0
    rec['user_id'] = user_id
    rec['name'] = name or rec.get('name', user_id)
    paths = list(rec.get('aligned_paths', [])) if isinstance(rec.get('aligned_paths', []), list) else []
    if aligned_rel and aligned_rel not in paths:
        paths.append(aligned_rel)
    rec['aligned_paths'] = paths
    persons[user_id] = rec
    meta['persons'] = persons
    meta['version'] = 2
    if 'labels' not in meta:
        meta['labels'] = []

# Delete: remove person record and aligned files if requested

def delete_person(idx: FaceIndex, user_id: str, index_path: str, labels_path: str, remove_files: bool=True) -> int:
    """Delete user vectors + metadata safely.

    Only rewrite the labels list if vectors were actually removed. This prevents
    wiping the labels array when the in-memory index failed to load (empty) but
    labels.json still contains other identities.
    """
    # Attempt vector removal
    try:
        removed = idx.remove_label(user_id)
    except Exception:
        removed = 0

    meta = read_labels(labels_path)
    if not isinstance(meta, dict):
        meta = {}
    persons = meta.get('persons', {}) if isinstance(meta.get('persons', {}), dict) else {}
    rec = persons.get(user_id)

    # Remove aligned image files only if we actually have a record and we want to remove files
    if remove_files and isinstance(rec, dict):
        for rel in rec.get('aligned_paths', []):
            try:
                abs_path = rel if os.path.isabs(rel) else os.path.join(os.getcwd(), rel)
                if os.path.exists(abs_path):
                    os.remove(abs_path)
            except Exception:
                pass

    # Remove person record from persons map regardless of whether vectors were present
    if isinstance(persons, dict) and user_id in persons:
        persons.pop(user_id, None)
    meta['persons'] = persons

    # If vectors were removed (index had this user), update labels list from index.
    # Otherwise preserve existing labels to avoid accidental clearing.
    if removed > 0:
        try:
            meta['labels'] = list(getattr(idx, '_labels', []))
        except Exception:
            pass
        # Write updated labels/persons first, then save index (so idx.save sees latest persons)
        write_labels(labels_path, meta)
        try:
            idx.save(index_path, labels_path)
        except Exception:
            pass
    else:
        # removed == 0; do NOT call idx.save (avoid overwriting labels with empty array)
        # Keep original labels array intact
        write_labels(labels_path, meta)
    return removed

# ---------------- Quality & Similarity Checks ---------------- #

def blur_variance(bgr) -> float:
    if cv2 is None or bgr is None:
        return 0.0
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0

def search_top(idx: FaceIndex, emb: np.ndarray) -> Tuple[Optional[str], float]:
    if idx is None or idx.size() <= 0:
        return None, -1.0
    try:
        name, score = idx.search_top1(emb)
        return name, float(score)
    except Exception:
        return None, -1.0

