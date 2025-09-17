import os
import configparser
import toml
import numpy as np

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


def safe_load_index(recog_cfg: dict):
    """Load a persisted FAISS index if available; do NOT build at runtime.

    Returns None when disabled, files missing, or FAISS unavailable.
    """
    try:
        dbg = os.getenv('DS_FAISS_DEBUG', '0') == '1'
        t0 = None
        if dbg:
            try:
                import time as _t
                t0 = _t.time()
                print('[FAISS] safe_load_index: start')
            except Exception:
                pass
        if not isinstance(recog_cfg, dict):
            return None
        # Allow hard-disable via env to bypass any FAISS work
        if os.getenv('DS_DISABLE_FAISS', '0') == '1':
            if dbg:
                print('[FAISS] disabled by env DS_DISABLE_FAISS=1')
            return None
        use_index = int(recog_cfg.get('use_index', 1))
        if use_index != 1:
            if dbg:
                print('[FAISS] disabled by recognition.use_index!=1')
            return None
        index_path = str(recog_cfg.get('index_path', '')).strip()
        labels_path = str(recog_cfg.get('labels_path', '')).strip()
        use_gpu = bool(int(recog_cfg.get('use_gpu', 0)))
        gpu_id = int(recog_cfg.get('gpu_id', 0))
        # Only attempt import if files exist to avoid unnecessary module import
        if index_path and labels_path and os.path.exists(index_path) and os.path.exists(labels_path):
            try:
                # Lazy import to avoid import-time hangs when FAISS is not healthy
                from utils.faiss_index import FaceIndex as _FaceIndex  # type: ignore
            except Exception as e:
                if dbg:
                    print(f'[FAISS] import failed; skipping: {e}')
                return None
            try:
                idx = _FaceIndex.load(index_path, labels_path, use_gpu=use_gpu, gpu_id=gpu_id)
                if dbg:
                    try:
                        import time as _t
                        dt = (_t.time() - t0) if t0 else -1
                        sz = -1
                        try:
                            sz = idx.size()
                        except Exception:
                            pass
                        print(f'[FAISS] loaded ok: size={sz}, gpu={int(use_gpu)}, dt={dt:.3f}s')
                    except Exception:
                        pass
                return idx
            except Exception as e:
                if dbg:
                    print(f'[FAISS] load failed; skipping: {e}')
                return None
        else:
            if dbg:
                print('[FAISS] files missing; skipping')
        return None
    except Exception:
        if os.getenv('DS_FAISS_DEBUG', '0') == '1':
            print('[FAISS] exception; skipping')
        return None