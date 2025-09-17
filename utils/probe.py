import os
import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

import pyds

def pgie_src_filter_probe(pad,info,u_data):
    """
    Probe to extract facial info from metadata and add them to Face pool. 
    
    Should be after retinaface.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    # Threshold from config with env override
    cfg_min = None
    try:
        if isinstance(u_data, dict):
            cfg_min = u_data.get('thresholds', {}).get('pgie_min_conf', None)
    except Exception:
        cfg_min = None
    env_min = os.getenv('PGIE_MIN_CONF')
    try:
        if env_min is not None:
            MIN_CONF = float(env_min)
        elif cfg_min is not None:
            MIN_CONF = float(cfg_min)
        else:
            MIN_CONF = 0.6
    except Exception:
        MIN_CONF = 0.6
    # One-time debug to confirm probe is attached and env is read
    if not hasattr(pgie_src_filter_probe, '_dbg_once'):
        pgie_src_filter_probe._dbg_once = True
        try:
            src = 'ENV' if env_min is not None else ('CFG' if cfg_min is not None else 'DEFAULT')
            print(f"[PGIE_PROBE] attached. MIN_CONF={MIN_CONF} (src={src})", flush=True)
        except Exception:
            pass

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        # Original simple per-object filtering based on confidence only
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            drop_signal = True
            try:
                conf = float(getattr(obj_meta, 'confidence', 0.0))
            except Exception:
                conf = 0.0
            if conf > MIN_CONF:
                drop_signal = False

            try:
                next_obj = l_obj.next
            except StopIteration:
                next_obj = None

            if drop_signal is True:
                pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)

            l_obj = next_obj

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def sgie_feature_extract_probe(pad,info, data):
    """
    Probe to extract facial feature from user-meta data. 
    
    Should be after arcface.
    """
    loaded_faces = data[0]
    threshold = 0.3
    if isinstance(data, (list, tuple)) and len(data) >= 4:
        try:
            threshold = float(data[3])
        except Exception:
            threshold = 0.3
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))    
    l_frame = batch_meta.frame_meta_list

    import cv2

    # Save mode control
    recog_save_dir = data[5] if len(data) > 5 and data[5] else ''
    recog_save_mode = (data[6] if len(data) > 6 and data[6] else 'all').lower()
    if recog_save_mode not in ('all','first','best','none'):
        recog_save_mode = 'all'
    # Ensure directories exist
    try:
        if recog_save_dir:
            os.makedirs(recog_save_dir, exist_ok=True)
    except Exception as _:
        pass

    # Track best/first saves per object id
    # We stash state on the probe function object to persist across calls
    if not hasattr(sgie_feature_extract_probe, '_track_state'):
        sgie_feature_extract_probe._track_state = {}
    track_state = sgie_feature_extract_probe._track_state
    # Verbosity and saving controls (from main -> data vector)
    verbose = False
    try:
        # index 7 reserved for verbose flag (bool)
        if isinstance(data, (list, tuple)) and len(data) > 7:
            verbose = bool(data[7])
    except Exception:
        verbose = False

    # Vector index (FAISS) and metric
    vector_index = None
    recog_metric = 'cosine'
    try:
        if isinstance(data, (list, tuple)) and len(data) > 8:
            vector_index = data[8]
        if isinstance(data, (list, tuple)) and len(data) > 9:
            recog_metric = str(data[9]).lower()
    except Exception:
        vector_index = None
        recog_metric = 'cosine'

    # One-time debug to confirm SGIE probe wiring and key params
    if not hasattr(sgie_feature_extract_probe, '_dbg_once'):
        sgie_feature_extract_probe._dbg_once = True
        try:
            vi = None
            metric = None
            if isinstance(data, (list, tuple)):
                vi = data[8] if len(data) > 8 else None
                metric = (data[9] if len(data) > 9 else 'cosine')
            visz = -1
            try:
                visz = vi.size() if vi is not None else 0
            except Exception:
                visz = 0
            print(
                "[SGIE_PROBE] attached. threshold=" + str(threshold) +
                ", save_dir=" + str(recog_save_dir) +
                ", save_mode=" + str(recog_save_mode) +
                ", vector_index_size=" + str(visz) +
                ", metric=" + str(metric),
                flush=True
            )
        except Exception:
            pass

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        frame_number = frame_meta.frame_num
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            face_feature = get_face_feature(obj_meta, frame_number, data)
            if face_feature is not None:
                top_name = None
                top_sim = -1.0  # higher is better similarity proxy
                top_dist = None # for l2 only
                # Prefer FAISS index if available
                if vector_index is not None and vector_index.size() > 0:
                    try:
                        name, score = vector_index.search_top1(face_feature.reshape(-1))
                        if recog_metric == 'l2':
                            top_name = name
                            top_dist = float(score)
                            top_sim = -top_dist
                        else:
                            top_name = name
                            top_sim = float(score)
                    except Exception:
                        top_name, top_sim, top_dist = None, -1.0, None
                elif loaded_faces:
                    emb = face_feature.reshape(-1)
                    if recog_metric == 'l2':
                        best = (None, float('inf'))
                        for key, value in loaded_faces.items():
                            diff = emb - value.reshape(-1)
                            dist = float(np.dot(diff, diff))
                            if dist < best[1]:
                                best = (key, dist)
                        if best[0] is not None:
                            top_name = best[0]
                            top_dist = best[1]
                            top_sim = -top_dist
                    else:
                        for key, value in loaded_faces.items():
                            score = float(np.dot(emb, value.reshape(-1)))
                            if score > top_sim:
                                top_sim = score
                                top_name = key
                # Decide match based on metric-specific threshold
                match_ok = False
                if top_name is not None:
                    if recog_metric == 'l2':
                        if top_dist is not None and top_dist <= threshold:
                            match_ok = True
                    else:
                        if top_sim >= threshold:
                            match_ok = True
                if match_ok:
                    if verbose:
                        disp = top_dist if recog_metric == 'l2' else top_sim
                        print(f"[SGIE_PROBE] frame-{frame_number} id-{obj_meta.object_id} match: {top_name} score: {disp:.3f}", flush=True)
                    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                    display_meta.num_labels = 1
                    py_nvosd_text_params = display_meta.text_params[0]
                    py_nvosd_text_params.display_text = top_name
                    py_nvosd_text_params.x_offset = int(obj_meta.rect_params.left)
                    py_nvosd_text_params.y_offset = int(obj_meta.rect_params.top + obj_meta.rect_params.height)
                    py_nvosd_text_params.font_params.font_name = "Serif"
                    py_nvosd_text_params.font_params.font_size = 20
                    py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                    py_nvosd_text_params.set_bg_clr = 1
                    py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
                    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

                    # --- Save recognized face image ---
                    # Decide if we should save this frame based on mode
                    try:
                        align_dir = data[4] if len(data) > 4 and data[4] else None
                        # Output dir: recognition.save_dir if provided, else fall back to features save_path
                        out_dir = recog_save_dir if recog_save_dir else (data[2] if len(data) > 2 and data[2] else '.')
                        os.makedirs(out_dir, exist_ok=True)

                        oid = int(obj_meta.object_id)
                        key = oid
                        st = track_state.get(key)
                        if st is None:
                            st = {'saved_first': False, 'best_score': -1.0, 'best_path': None, 'best_name': None, 'best_frame': None}
                            track_state[key] = st

                        def copy_aligned(dst_dir):
                            if not align_dir:
                                return None
                            src = os.path.join(align_dir, f"frame-{frame_number}_object-{oid}-aligned.png")
                            if os.path.exists(src):
                                import shutil
                                # Use stable name for best mode, per-frame name for others
                                if recog_save_mode == 'best':
                                    dst = os.path.join(dst_dir, f"recog_{top_name}_track{oid}_best.png")
                                else:
                                    dst = os.path.join(dst_dir, f"recog_{top_name}_f{frame_number}_id{oid}.png")
                                shutil.copy(src, dst)
                                return dst
                            return None

                        if recog_save_mode == 'none':
                            pass
                        elif recog_save_mode == 'all':
                            _ = copy_aligned(out_dir)
                        elif recog_save_mode == 'first':
                            if not st['saved_first']:
                                dst = copy_aligned(out_dir)
                                if dst:
                                    st['saved_first'] = True
                        else:  # best
                            if top_sim > st['best_score']:
                                dst = copy_aligned(out_dir)
                                if dst:
                                    st['best_score'] = top_sim
                                    st['best_path'] = dst
                                    st['best_name'] = top_name
                                    st['best_frame'] = frame_number
                    except Exception as e:
                        if verbose:
                            print(f"[WARN] Could not save recognized face image: {e}", flush=True)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def get_face_feature(obj_meta, frame_num, data):
    """Get face feature from user-meta data.
    
    Args:
        obj_meta (NvDsObjectMeta): Object metadata.
    Returns:
        np.array: Normalized face feature.
    """
    l_user_meta = obj_meta.obj_user_meta_list
    while l_user_meta:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data) 
        except StopIteration:
            break
        if user_meta and user_meta.base_meta.meta_type==pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META: 
            try:
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
            except StopIteration:
                break
    
            layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
            output = []
            for i in range(512):
                output.append(pyds.get_detections(layer.buffer, i))
            res = np.reshape(output,(1,-1))
            norm=np.linalg.norm(res)                    
            normal_array = res / norm
            # Save feature .npy only if enabled
            try:
                if data[1]:
                    save_p = os.path.join(data[2], f"{obj_meta.object_id}-{frame_num}.npy")
                    np.save(save_p, normal_array)
                    try:
                        print(f"[SGIE_PROBE] saved feature: {save_p}", flush=True)
                    except Exception:
                        pass
            except Exception:
                pass
            return normal_array

        try:
            l_user_meta = l_user_meta.next
        except StopIteration:
            break

    return None