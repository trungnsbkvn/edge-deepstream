'''
Author: zhouyuchong
Date: 2024-08-19 14:54:53
Description: 
LastEditors: zhouyuchong
LastEditTime: 2024-08-20 10:03:25
'''
import os
import configparser
import toml
import numpy as np

# Optional FAISS index
try:
    from utils.faiss_index import FaceIndex, FaceIndexConfig
except Exception:
    FaceIndex = None
    FaceIndexConfig = None
def parse_args(cfg_path):
    cfg = toml.load(cfg_path)
    return cfg

def set_property(cfg, gst_element, name):
    properties = cfg[name]
    for key, value in properties.items():
        print(f"{name} set_property {key} {value} \n")
        gst_element.set_property(key, value)

def set_tracker_properties(tracker, path):
    config = configparser.ConfigParser()
    config.read(path)
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)


def load_faces(path):
    loaded_faces = {}
    file_list = os.listdir(path)
    for file in file_list:
        if file.endswith('.npy'):
            face_feature = np.load(os.path.join(path, file)).reshape(-1, 1)
            name = file.split('.')[0]
            loaded_faces[name] = face_feature
    return loaded_faces


def build_or_load_index(known_face_dir: str, recog_cfg: dict):
    """Build or load a FAISS-based index from known faces directory using config.

    Returns None if FAISS not available or use_index is disabled.
    """
    use_index = int(recog_cfg.get('use_index', 1)) if isinstance(recog_cfg, dict) else 1
    if not use_index:
        return None
    if FaceIndex is None or FaceIndexConfig is None:
        # FAISS not installed; fallback to Python matching
        return None

    metric = str(recog_cfg.get('metric', 'cosine'))
    index_type = str(recog_cfg.get('index_type', 'flat'))
    use_gpu = bool(recog_cfg.get('use_gpu', 1))
    gpu_id = int(recog_cfg.get('gpu_id', 0))
    nlist = int(recog_cfg.get('nlist', 0)) or None
    m_pq = int(recog_cfg.get('m_pq', 0)) or None
    nbits_pq = int(recog_cfg.get('nbits_pq', 8))
    index_path = str(recog_cfg.get('index_path', ''))
    labels_path = str(recog_cfg.get('labels_path', ''))
    feature_dir = str(recog_cfg.get('feature_dir', '')).strip() or known_face_dir

    cfg = FaceIndexConfig(
        metric=metric,
        index_type=index_type,
        use_gpu=use_gpu,
        gpu_id=gpu_id,
        nlist=nlist,
        m_pq=m_pq,
        nbits_pq=nbits_pq,
    )

    # Try load existing index if both files exist
    if index_path and labels_path and os.path.exists(index_path) and os.path.exists(labels_path):
        try:
            return FaceIndex.load(index_path, labels_path, use_gpu=use_gpu, gpu_id=gpu_id)
        except Exception:
            pass

    # Else build from directory and save
    try:
        idx = FaceIndex.from_dir(feature_dir, cfg)
        if index_path and labels_path:
            try:
                os.makedirs(os.path.dirname(index_path), exist_ok=True)
                os.makedirs(os.path.dirname(labels_path), exist_ok=True)
                idx.save(index_path, labels_path)
            except Exception:
                pass
        return idx
    except Exception:
        return None