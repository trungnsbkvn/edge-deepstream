"""
Lightweight vector index wrapper for face embeddings using FAISS (GPU if available),
with optional cuVS backend (if installed) for IVF/PQ. Designed for cosine/IP search.

Public API:
 - FaceIndex.from_dir(dir_path, metric='cosine', index_type='flat', use_gpu=True, **kwargs)
 - FaceIndex.load(index_path, labels_path, use_gpu=True)
 - add(names, vectors)
 - search_top1(vector) -> (name, score)
 - search(vectors, k=5) -> (names_list, scores)
 - save(index_path, labels_path)

Notes:
 - Assumes input vectors are L2-normalized when using cosine/IP.
 - For persistence, we always save a CPU FAISS index; GPU index is reconstructed on load if requested.
"""
from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

import numpy as np


def _try_import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except Exception:
        return None


def _try_import_cuvs():
    # Optional cuVS (RAPIDS) backend. We keep import soft to avoid runtime errors.
    try:
        import cuvs  # type: ignore
        return cuvs
    except Exception:
        return None


_faiss = _try_import_faiss()
_cuvs = _try_import_cuvs()


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 1:
        return a.reshape(1, -1)
    if a.ndim == 2 and a.shape[1] == 1:
        return a.reshape(1, -1)
    return a


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


@dataclass
class FaceIndexConfig:
    metric: str = "cosine"  # 'cosine' or 'l2' (cosine uses inner-product)
    index_type: str = "flat"  # 'flat' | 'ivf' | 'ivfpq'
    use_gpu: bool = True
    gpu_id: int = 0
    nlist: Optional[int] = None  # IVF list count; if None, auto from sqrt(N)
    m_pq: Optional[int] = None   # PQ subvectors for IVFPQ
    nbits_pq: int = 8            # bits per codebook entry


class FaceIndex:
    def __init__(self, dim: int, cfg: Optional[FaceIndexConfig] = None):
        if _faiss is None:
            raise RuntimeError("FAISS is not installed. Please install faiss-cpu or faiss-gpu.")
        self.dim = int(dim)
        self.cfg = cfg or FaceIndexConfig()
        self._cpu_index = None
        self._gpu_index = None
        self._gpu_res = None
        self._labels: List[str] = []  # id -> name
        self._backend = "faiss"  # future: 'cuvs'

    # ---------- construction ----------
    @staticmethod
    def from_dir(dir_path: str, cfg: Optional[FaceIndexConfig] = None) -> "FaceIndex":
        # Load .npy features from dir; file stem is label
        file_list = [f for f in os.listdir(dir_path) if f.lower().endswith(".npy")]
        if not file_list:
            raise FileNotFoundError(f"No .npy features found in {dir_path}")
        labels: List[str] = []
        vecs: List[np.ndarray] = []
        for f in sorted(file_list):
            try:
                v = np.load(os.path.join(dir_path, f))
                v = np.asarray(v).reshape(-1)
                vecs.append(v)
                labels.append(os.path.splitext(f)[0])
            except Exception:
                # Skip unreadable/corrupt entries
                continue
        if not vecs:
            raise ValueError(f"No valid vectors loaded from {dir_path}")
        X = np.vstack(vecs).astype(np.float32)
        # For cosine metric we want normalized embeddings (ArcFace already normalized, but ensure anyway)
        if (cfg.metric if cfg else "cosine") == "cosine":
            X = _normalize_rows(X)
        idx = FaceIndex(dim=X.shape[1], cfg=cfg)
        idx.build(X, labels)
        return idx

    def _make_faiss_index(self, nvecs: int):
        metric = self.cfg.metric.lower()
        use_ip = metric in ("cosine", "ip", "inner", "inner_product")
        metric_type = _faiss.METRIC_INNER_PRODUCT if use_ip else _faiss.METRIC_L2

        index_type = self.cfg.index_type.lower()
        if index_type == "flat":
            cpu_index = _faiss.IndexFlat(self.dim, metric_type)
            needs_train = False
        elif index_type in ("ivf", "ivfflat"):
            nlist = self.cfg.nlist or max(1, int(math.sqrt(max(1, nvecs))))
            quantizer = _faiss.IndexFlat(self.dim, metric_type)
            cpu_index = _faiss.IndexIVFFlat(quantizer, self.dim, nlist, metric_type)
            needs_train = True
        elif index_type in ("ivfpq", "pq"):
            nlist = self.cfg.nlist or max(1, int(math.sqrt(max(1, nvecs))))
            m = self.cfg.m_pq or max(1, self.dim // 8)
            quantizer = _faiss.IndexFlat(self.dim, metric_type)
            cpu_index = _faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, self.cfg.nbits_pq, metric_type)
            needs_train = True
        else:
            raise ValueError(f"Unsupported index_type: {self.cfg.index_type}")
        return cpu_index, needs_train

    def _to_gpu(self, cpu_index):
        if not self.cfg.use_gpu:
            return None
        try:
            ngpu = _faiss.get_num_gpus()
        except Exception:
            return None
        if ngpu is None or ngpu <= 0:
            return None
        try:
            res = _faiss.StandardGpuResources()
            gpu_idx = int(self.cfg.gpu_id) if hasattr(self.cfg, 'gpu_id') else 0
            gpu_index = _faiss.index_cpu_to_gpu(res, gpu_idx, cpu_index)
            self._gpu_res = res
            return gpu_index
        except Exception:
            return None

    def build(self, X: np.ndarray, labels: Sequence[str]):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            X = X.reshape(len(X), -1)
        if self.cfg.metric.lower() == "cosine":
            X = _normalize_rows(X)
        n, d = X.shape
        if d != self.dim:
            raise ValueError(f"Dim mismatch: got {d}, expected {self.dim}")
        self._labels = list(labels)
        cpu_index, needs_train = self._make_faiss_index(n)
        if needs_train:
            cpu_index.train(X)
        cpu_index.add(X)
        self._cpu_index = cpu_index
        self._gpu_index = self._to_gpu(cpu_index)

    # ---------- mutation ----------
    def add(self, names: Sequence[str], vectors: np.ndarray):
        V = _ensure_2d(np.asarray(vectors, dtype=np.float32))
        if self.cfg.metric.lower() == "cosine":
            V = _normalize_rows(V)
        if V.shape[1] != self.dim:
            raise ValueError("Vector dim mismatch")
        # Always add to CPU index for persistence; refresh GPU from CPU
        if self._cpu_index is None:
            self._cpu_index, _ = self._make_faiss_index(len(names))
        self._cpu_index.add(V)
        self._labels.extend(list(names))
        # Rebuild GPU mirror if present
        if self._gpu_index is not None:
            try:
                self._gpu_index = self._to_gpu(self._cpu_index)
            except Exception:
                self._gpu_index = None

    # ---------- query ----------
    def search(self, vectors: np.ndarray, k: int = 5) -> Tuple[List[List[str]], np.ndarray]:
        V = _ensure_2d(np.asarray(vectors, dtype=np.float32))
        if self.cfg.metric.lower() == "cosine":
            V = _normalize_rows(V)
        index = self._gpu_index or self._cpu_index
        if index is None:
            raise RuntimeError("Index is empty; build or load before search.")
        D, I = index.search(V, k)
        names = [[self._labels[idx] if 0 <= idx < len(self._labels) else "" for idx in row] for row in I]
        return names, D

    def search_top1(self, vector: np.ndarray) -> Tuple[Optional[str], float]:
        names, scores = self.search(vector, k=1)
        if names and names[0]:
            return names[0][0], float(scores[0][0])
        return None, float("-inf")

    # ---------- persistence ----------
    def save(self, index_path: str, labels_path: str):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(labels_path), exist_ok=True)
        cpu_index = self._cpu_index
        if cpu_index is None and self._gpu_index is not None:
            # move GPU -> CPU for saving
            cpu_index = _faiss.index_gpu_to_cpu(self._gpu_index)
        if cpu_index is None:
            raise RuntimeError("No index to save")
        _faiss.write_index(cpu_index, index_path)
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump({"labels": self._labels}, f, ensure_ascii=False)

    @staticmethod
    def load(index_path: str, labels_path: str, use_gpu: bool = True, gpu_id: int = 0) -> "FaceIndex":
        if _faiss is None:
            raise RuntimeError("FAISS is not installed.")
        if not os.path.exists(index_path):
            raise FileNotFoundError(index_path)
        if not os.path.exists(labels_path):
            raise FileNotFoundError(labels_path)
        cpu_index = _faiss.read_index(index_path)
        dim = cpu_index.d
        cfg = FaceIndexConfig(use_gpu=use_gpu, gpu_id=gpu_id)
        idx = FaceIndex(dim=dim, cfg=cfg)
        idx._cpu_index = cpu_index
        idx._gpu_index = idx._to_gpu(cpu_index)
        with open(labels_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        idx._labels = list(meta.get("labels", []))
        return idx

    # ---------- utils ----------
    def size(self) -> int:
        if self._cpu_index is None:
            return 0
        try:
            return self._cpu_index.ntotal
        except Exception:
            return len(self._labels)


def ensure_parent(path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
