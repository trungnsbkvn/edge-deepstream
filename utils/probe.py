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
    
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:

                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            
            drop_signal = True
            if obj_meta.confidence > 0.6:
                drop_signal = False

            try: 
                l_obj=l_obj.next
                if drop_signal is True:
                    pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
                
            except StopIteration:
                break

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
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        l_obj=frame_meta.obj_meta_list
        frame_number = frame_meta.frame_num
        while l_obj is not None:
            try:
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                
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
                                  
            try: 
                l_obj=l_obj.next

            except StopIteration:
                break  
        try:
            l_frame=l_frame.next
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
            if data[1]:
                # print(data[1], data[2])
                save_p = os.path.join(data[2], f"{obj_meta.object_id}-{frame_num}.npy")
                np.save(save_p, normal_array)
            return normal_array

        try:
            l_user_meta = l_user_meta.next
        except StopIteration:
            break

    return None