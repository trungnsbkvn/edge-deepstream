/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>

#include "nvdsinfer_custom_impl.h"

// #include "utils.h"
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLIP(a, min, max) (MAX(MIN(a, max), min))

#define NMS_THRESH 0.45;

extern "C" bool
NvDsInferParseYoloFace(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferObjectDetectionInfo>& objectList);

static std::vector<NvDsInferObjectDetectionInfo>
nonMaximumSuppression(std::vector<NvDsInferObjectDetectionInfo> binfo)
{
  auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
    if (x1min > x2min) {
      std::swap(x1min, x2min);
      std::swap(x1max, x2max);
    }
    return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
  };

  auto computeIoU = [&overlap1D](NvDsInferObjectDetectionInfo& bbox1, NvDsInferObjectDetectionInfo& bbox2) -> float {
    float overlapX = overlap1D(bbox1.left, bbox1.left + bbox1.width, bbox2.left, bbox2.left + bbox2.width);
    float overlapY = overlap1D(bbox1.top, bbox1.top + bbox1.height, bbox2.top, bbox2.top + bbox2.height);
    float area1 = (bbox1.width) * (bbox1.height);
    float area2 = (bbox2.width) * (bbox2.height);
    float overlap2D = overlapX * overlapY;
    float u = area1 + area2 - overlap2D;
    return u == 0 ? 0 : overlap2D / u;
  };

  std::stable_sort(binfo.begin(), binfo.end(), [](const NvDsInferObjectDetectionInfo& b1, const NvDsInferObjectDetectionInfo& b2) {
    return b1.detectionConfidence > b2.detectionConfidence;
  });

  std::vector<NvDsInferObjectDetectionInfo> out;
  for (auto i : binfo) {
    bool keep = true;
    for (auto j : out) {
      if (keep) {
        float overlap = computeIoU(i, j);
        keep = overlap <= NMS_THRESH;
      }
      else {
        break;
      }
    }
    if (keep) {
      out.push_back(i);
    }
  }
  return out;
}

static std::vector<NvDsInferObjectDetectionInfo>
nmsAllClasses(std::vector<NvDsInferObjectDetectionInfo>& binfo)
{
  std::vector<NvDsInferObjectDetectionInfo> result = nonMaximumSuppression(binfo);
  return result;
}

// static void
// addFaceProposal(const float* landmarks, const uint& landmarksSizeRaw, const uint& netW, const uint& netH, const uint& b,
//     NvDsInferObjectDetectionInfo& bbi)
// {
//   uint landmarksSize = landmarksSizeRaw == 10 ? landmarksSizeRaw + 5 : landmarksSizeRaw;
//   bbi.mask = new float[landmarksSize];
//   for (uint p = 0; p < landmarksSize / 3; ++p) {
//     if (landmarksSizeRaw == 10) {
//       bbi.mask[p * 3 + 0] = clamp(landmarks[b * landmarksSizeRaw + p * 2 + 0], 0, netW);
//       bbi.mask[p * 3 + 1] = clamp(landmarks[b * landmarksSizeRaw + p * 2 + 1], 0, netH);
//       bbi.mask[p * 3 + 2] = 1.0;
//     }
//     else {
//       bbi.mask[p * 3 + 0] = clamp(landmarks[b * landmarksSize + p * 3 + 0], 0, netW);
//       bbi.mask[p * 3 + 1] = clamp(landmarks[b * landmarksSize + p * 3 + 1], 0, netH);
//       bbi.mask[p * 3 + 2] = landmarks[b * landmarksSize + p * 3 + 2];
//     }
//   }
//   bbi.mask_width = netW;
//   bbi.mask_height = netH;
//   bbi.mask_size = sizeof(float) * landmarksSize;
// }

static NvDsInferObjectDetectionInfo
convertBBox(const float& bx1, const float& by1, const float& bx2, const float& by2, const uint& netW, const uint& netH)
{
  NvDsInferObjectDetectionInfo b;

  float x1 = bx1;
  float y1 = by1;
  float x2 = bx2;
  float y2 = by2;

  x1 = CLIP(x1, 0, netW);
  y1 = CLIP(y1, 0, netH);
  x2 = CLIP(x2, 0, netW);
  y2 = CLIP(y2, 0, netH);

  b.left = x1;
  b.width = CLIP(x2 - x1, 0, netW);
  b.top = y1;
  b.height = CLIP(y2 - y1, 0, netH);

  return b;
}

static bool
addBBoxProposal(const float bx1, const float by1, const float bx2, const float by2, const uint& netW, const uint& netH,
    const int maxIndex, const float maxProb, NvDsInferObjectDetectionInfo& bbi)
{
  bbi = convertBBox(bx1, by1, bx2, by2, netW, netH);

  if (bbi.width < 1 || bbi.height < 1) {
      return false;
  }

  bbi.detectionConfidence = maxProb;
  bbi.classId = maxIndex;
  return true;
}

static std::vector<NvDsInferObjectDetectionInfo>
decodeTensorYoloFace(const float* boxes, const float* scores, const float* landmarks, const uint& outputSize,
    const uint& landmarksSize, const uint& netW, const uint& netH, const std::vector<float>& preclusterThreshold)
{
  std::vector<NvDsInferObjectDetectionInfo> binfo;

  for (uint b = 0; b < outputSize; ++b) {
    float maxProb = scores[b];

    if (maxProb < preclusterThreshold[0]) {
      continue;
    }

    float bxc = boxes[b * 4 + 0];
    float byc = boxes[b * 4 + 1];
    float bw = boxes[b * 4 + 2];
    float bh = boxes[b * 4 + 3];

    float bx1 = bxc - bw / 2;
    float by1 = byc - bh / 2;
    float bx2 = bx1 + bw;
    float by2 = by1 + bh;

    NvDsInferObjectDetectionInfo bbi;

  if (!addBBoxProposal(bx1, by1, bx2, by2, netW, netH, 0, maxProb, bbi)) {
      continue;
    }

    // Landmarks layout can be either 10 (x,y pairs) or 15 (x,y,score per point)
    unsigned int num_points = 5;
    if (landmarks && (landmarksSize == 10 || landmarksSize >= 15)) {
      for (unsigned int i2 = 0; i2 < num_points; i2++) {
        unsigned int base = b * landmarksSize;
        unsigned int x_idx = (landmarksSize == 10) ? (base + i2 * 2) : (base + i2 * 3);
        unsigned int y_idx = (landmarksSize == 10) ? (base + i2 * 2 + 1) : (base + i2 * 3 + 1);
        float lx = landmarks[x_idx];
        float ly = landmarks[y_idx];
        // If the model outputs normalized coords (0..1), scale to pixels
        if (lx >= 0.0f && lx <= 1.5f && ly >= 0.0f && ly <= 1.5f) {
          lx *= static_cast<float>(netW);
          ly *= static_cast<float>(netH);
        }
        // Basic sanity: guard NaN/Inf
        if (!std::isfinite(lx) || !std::isfinite(ly)) {
          lx = (bx1 + bx2) * 0.5f; // fallback to box center
          ly = (by1 + by2) * 0.5f;
        }
        bbi.landmark[i2 * 2] = lround(CLIP(lx, 0, netW));
        bbi.landmark[i2 * 2 + 1] = lround(CLIP(ly, 0, netH));
      }
  bbi.numLmks = 10;
    } else {
      // Fallback: synthesize 5 points from bbox to avoid bad alignment crashes
      float cx = (bx1 + bx2) * 0.5f;
      float cy = (by1 + by2) * 0.5f;
      float w = MAX(1.0f, bx2 - bx1);
      float h = MAX(1.0f, by2 - by1);
      float lx[5] = {cx - 0.2f * w, cx + 0.2f * w, cx, cx - 0.15f * w, cx + 0.15f * w};
      float ly[5] = {cy - 0.2f * h, cy - 0.2f * h, cy, cy + 0.2f * h, cy + 0.2f * h};
      for (unsigned int i2 = 0; i2 < num_points; i2++) {
        bbi.landmark[i2 * 2] = lround(CLIP(lx[i2], 0, netW));
        bbi.landmark[i2 * 2 + 1] = lround(CLIP(ly[i2], 0, netH));
      }
      bbi.numLmks = 10;
    }

    // Additional safety: ensure bbox is large enough and landmarks lie within bbox
    const float min_box_size = 12.0f;
    if (bbi.width < min_box_size || bbi.height < min_box_size) {
      continue;
    }
    bool lmk_valid = true;
    for (unsigned int i2 = 0; i2 < 5; i2++) {
      int lx = bbi.landmark[i2 * 2];
      int ly = bbi.landmark[i2 * 2 + 1];
      if (lx < bbi.left - 2 || lx > (bbi.left + bbi.width + 2) ||
          ly < bbi.top - 2 || ly > (bbi.top + bbi.height + 2)) {
        lmk_valid = false;
        break;
      }
    }
    // Additional geometry checks for 5-point order: [LE, RE, Nose, LM, RM]
    if (lmk_valid) {
      auto dx = [](int x1, int y1, int x2, int y2) {
        float dx = static_cast<float>(x2 - x1);
        float dy = static_cast<float>(y2 - y1);
        return std::sqrt(dx * dx + dy * dy);
      };
      int le_x = bbi.landmark[0], le_y = bbi.landmark[1];
      int re_x = bbi.landmark[2], re_y = bbi.landmark[3];
      int n_x  = bbi.landmark[4], n_y  = bbi.landmark[5];
      int lm_x = bbi.landmark[6], lm_y = bbi.landmark[7];
      int rm_x = bbi.landmark[8], rm_y = bbi.landmark[9];
      float eye_dist = dx(le_x, le_y, re_x, re_y);
      float box_diag = std::sqrt(bbi.width * bbi.width + bbi.height * bbi.height);
      if (eye_dist < 0.05f * box_diag || eye_dist > 1.2f * box_diag) {
        lmk_valid = false;
      }
      // Eyes should be roughly above mouth points
      if (!(le_y < lm_y && re_y < rm_y)) {
        lmk_valid = false;
      }
    }
    if (!lmk_valid) {
      continue;
    }

    binfo.push_back(bbi);
  }

  return binfo;
}

static bool NvDsInferParseCustomYoloFace(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                    NvDsInferNetworkInfo const &networkInfo,
                                    NvDsInferParseDetectionParams const &detectionParams,
                                    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo& boxes = outputLayersInfo[0];
  const NvDsInferLayerInfo& scores = outputLayersInfo[1];
  const NvDsInferLayerInfo& landmarks = outputLayersInfo[2];

  const uint outputSize = boxes.inferDims.d[0];
  const uint landmarksSize = landmarks.inferDims.d[1];

  std::vector<NvDsInferObjectDetectionInfo> objects = decodeTensorYoloFace((const float*) (boxes.buffer),
      (const float*) (scores.buffer), (const float*) (landmarks.buffer), outputSize, landmarksSize, networkInfo.width,
      networkInfo.height, detectionParams.perClassPreclusterThreshold);
  // std::cout<<"objects size: "<<objects.size()<<std::endl;
  objectList.clear();
  objectList = nmsAllClasses(objects);
  // std::cout<<"objects size after nms: "<<objectList.size()<<std::endl;
  // for (auto r : objectList) {
  //   std::cout << "bbox: " << r.left << " " << r.top << " " << r.width << " " << r.height << " " << r.detectionConfidence << " " << r.classId << std::endl;
  // }

  return true;
}

extern "C" bool
NvDsInferParseYoloFace(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustomYoloFace(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloFace);