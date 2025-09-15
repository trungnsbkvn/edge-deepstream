'''
Author: zhouyuchong
Date: 2024-08-19 14:35:17
Description: 
LastEditors: zhouyuchong
LastEditTime: 2024-09-19 14:36:41
'''
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
    
    # Tuning knobs: tighten to reduce noisy crops
    MIN_CONF = float(os.getenv('PGIE_MIN_CONF', '0.65'))   # raise if still noisy
    MIN_W = int(os.getenv('PGIE_MIN_W', '20'))            # min face width in pixels
    MIN_H = int(os.getenv('PGIE_MIN_H', '20'))            # min face height in pixels
    MIN_AR = float(os.getenv('PGIE_MIN_AR', '0.6'))       # w/h lower bound
    MAX_AR = float(os.getenv('PGIE_MAX_AR', '1.8'))       # w/h upper bound
    TOPK = int(os.getenv('PGIE_TOPK', '30'))              # keep top-K by conf per frame (0 disables)

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        # Collect objects first to rank by confidence
        objs = []
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            objs.append(obj_meta)
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Sort by confidence desc and keep top-K
        if objs:
            objs.sort(key=lambda o: float(getattr(o, 'confidence', 0.0)), reverse=True)
            keep_set = set(objs[:TOPK] if TOPK > 0 else objs)
        else:
            keep_set = set()

        # Second pass: remove low-conf, tiny, bad aspect, and extra beyond top-K
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            drop = False
            conf = float(getattr(obj_meta, 'confidence', 0.0))
            rp = obj_meta.rect_params
            w = float(getattr(rp, 'width', 0.0))
            h = float(getattr(rp, 'height', 0.0))
            ar = (w / h) if h > 1e-3 else 999.0

            if TOPK > 0 and obj_meta not in keep_set:
                drop = True
            elif conf < MIN_CONF:
                drop = True
            elif w < MIN_W or h < MIN_H:
                drop = True
            elif not (MIN_AR <= ar <= MAX_AR):
                drop = True

            try:
                next_obj = l_obj.next
            except StopIteration:
                next_obj = None

            if drop:
                pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)

            l_obj = next_obj

        try:
            l_frame=l_frame.next
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
            if face_feature is not None and loaded_faces:
                # Find top-1 match
                top_name = None
                top_score = -1.0
                for key, value in loaded_faces.items():
                    score = float(np.dot(face_feature, value)[0])
                    if score > top_score:
                        top_score = score
                        top_name = key
                # Print and overlay only if above threshold
                if top_score >= threshold and top_name is not None:
                    if verbose:
                        print(f"frame-{frame_number}, face-{obj_meta.object_id} match: {top_name} score: {top_score:.3f}")
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
                            if top_score > st['best_score']:
                                dst = copy_aligned(out_dir)
                                if dst:
                                    st['best_score'] = top_score
                                    st['best_path'] = dst
                                    st['best_name'] = top_name
                                    st['best_frame'] = frame_number
                    except Exception as e:
                        if verbose:
                            print(f"[WARN] Could not save recognized face image: {e}")

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
            except Exception:
                pass
            return normal_array

        try:
            l_user_meta = l_user_meta.next
        except StopIteration:
            break

    return None