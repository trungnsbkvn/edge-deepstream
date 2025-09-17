cmake_minimum_required(VERSION 3.10)
project(roi_helper LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# DeepStream paths (adjust if needed)
set(DS_ROOT /opt/nvidia/deepstream/deepstream)
include_directories(
  ${DS_ROOT}/sources/includes
  ${DS_ROOT}/sources/gst-plugins/gst-nvdssr
  ${DS_ROOT}/sources/gst-plugins/gst-nvvideo4linux2
  /usr/local/cuda/include
)

add_library(roi_helper SHARED roi_helper.cpp)

target_link_libraries(roi_helper
  nvbufsurface
  nvbufsurftransform
)

set_target_properties(roi_helper PROPERTIES OUTPUT_NAME "roi_helper")
