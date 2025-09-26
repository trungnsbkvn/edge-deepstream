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

# Find GStreamer and GLib via pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GLIB2 REQUIRED glib-2.0)
pkg_check_modules(GOBJECT REQUIRED gobject-2.0)

include_directories(
  ${GSTREAMER_INCLUDE_DIRS}
  ${GLIB2_INCLUDE_DIRS}
  ${GOBJECT_INCLUDE_DIRS}
)
link_directories(
  ${GSTREAMER_LIBRARY_DIRS}
  ${GLIB2_LIBRARY_DIRS}
  ${GOBJECT_LIBRARY_DIRS}
)
add_definitions(
  ${GSTREAMER_CFLAGS_OTHER}
  ${GLIB2_CFLAGS_OTHER}
  ${GOBJECT_CFLAGS_OTHER}
)

add_library(roi_helper SHARED roi_helper.cpp)

# Locate DeepStream libs
find_library(NVBUFSURFACE NAMES nvbufsurface HINTS ${DS_ROOT}/lib ${DS_ROOT}/lib/aarch64-linux-gnu)
find_library(NVBUFSURFTRANSFORM NAMES nvbufsurftransform HINTS ${DS_ROOT}/lib ${DS_ROOT}/lib/aarch64-linux-gnu)

message(STATUS "NVBUFSURFACE: ${NVBUFSURFACE}")
message(STATUS "NVBUFSURFTRANSFORM: ${NVBUFSURFTRANSFORM}")

target_link_libraries(roi_helper
  ${NVBUFSURFACE}
  ${NVBUFSURFTRANSFORM}
  ${GSTREAMER_LIBRARIES}
  ${GLIB2_LIBRARIES}
  ${GOBJECT_LIBRARIES}
)

set_target_properties(roi_helper PROPERTIES OUTPUT_NAME "roi_helper")
