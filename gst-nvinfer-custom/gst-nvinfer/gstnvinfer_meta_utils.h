/*
 * @Author: zhouyuchong
 * @Date: 2024-03-04 15:02:38
 * @Description: 
 * @LastEditors: zhouyuchong
 * @LastEditTime: 2024-09-14 17:12:20
 */
/**
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#include "gstnvinfer.h"
#include "gstnvinfer_impl.h"

/** set the user metadata type */
#define NVDS_USER_OBJECT_META_EXAMPLE (nvds_get_user_meta_type("NVIDIA.NVINFER.USER_META"))
/** custom meta to hold aligned image path (char*) */
#define NVDS_USER_OBJECT_META_ALIGNED_IMAGE_PATH (nvds_get_user_meta_type("NVIDIA.NVINFER.ALIGNED_IMAGE_PATH"))

void attach_metadata_detector (GstNvInfer * nvinfer, GstMiniObject * tensor_out_object,
        GstNvInferFrame & frame, NvDsInferDetectionOutput & detection_output,
        float segmentationThreshold);

void attach_metadata_classifier (GstNvInfer * nvinfer, GstMiniObject * tensor_out_object,
        GstNvInferFrame & frame, GstNvInferObjectInfo & object_info);

void merge_classification_output (GstNvInferObjectHistory & history,
    GstNvInferObjectInfo  &new_result);

void attach_metadata_segmentation (GstNvInfer * nvinfer, GstMiniObject * tensor_out_object,
        GstNvInferFrame & frame, NvDsInferSegmentationOutput & segmentation_output);

/* Attaches the raw tensor output to the GstBuffer as metadata. */
void attach_tensor_output_meta (GstNvInfer *nvinfer, GstMiniObject * tensor_out_object,
        GstNvInferBatch *batch, NvDsInferContextBatchOutput *batch_output);

/* Attach a string path (aligned image path) as user meta to an object. */
void attach_aligned_image_path_meta (GstNvInfer * nvinfer, NvDsObjectMeta *obj_meta, const char *path_str);
