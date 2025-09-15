################################################################################
# Copyright (c) 2017-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#################################################################################

CUDA_VER:=11.4
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

NVDS_VERSION:=6.3
ifeq ($(NVDS_VERSION),)
  $(error "NVDS_VERSION is not set")
endif

CXX:= g++
NVCC?= /usr/local/cuda-$(CUDA_VER)/bin/nvcc
SRCS:= gstnvinfer.cpp  gstnvinfer_allocator.cpp gstnvinfer_property_parser.cpp \
	gstnvinfer_meta_utils.cpp gstnvinfer_impl.cpp gstnvinfer_yaml_parser.cpp \
	   align_functions.cpp svd_rank_cuda.cpp svd_cuda_full.cpp
CU_SRCS:= align_functions_cuda.cu elementwise_minus_cuda.cu var_axis0_cuda.cu \
	   matrix_ops_cuda.cu matrix_ops_cuda_3x3.cu warp_affine_cuda.cu
INCS:= $(wildcard *.h)
LIB:=libnvdsgst_infer.so


CFLAGS+= -fPIC -std=c++11 -DDS_VERSION=\"$(NVDS_VERSION)\" \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/includes \
	 -I /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/gst-plugins/gst-nvdspreprocess/include \
	 -I /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/libs/nvdsinfer -DNDEBUG

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

LIBS := -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lcusolver -lcublas

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
       -lnvds_infer -lnvbufsurface -lnvbufsurftransform -ldl -lpthread -lyaml-cpp \
       -lcuda -Wl,-rpath,$(LIB_INSTALL_DIR) \
	   -lnppc  -lnppig -lnppicc -lnppial -lnppidei

OBJS:= $(SRCS:.cpp=.o) $(CU_SRCS:.cu=.o)

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 /usr/lib/aarch64-linux-gnu/pkgconfig/opencv4.pc
CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

# Test binary to compare CPU vs GPU alignment (disabled)
# TEST_BIN:=test_align_compare
# TEST_SRC:=tests/test_align_compare.cpp
# TEST_OBJS:=$(TEST_SRC:.cpp=.o)
# LINK_LIBS := $(filter-out -shared,$(LIBS))
# LINK_LIBS := $(filter-out -Wl,-no-undefined,$(LINK_LIBS))
# TEST_LIBS := \
# 	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lcusolver -lcublas \
# 	-L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta -lnvds_infer -lnvbufsurface -lnvbufsurftransform -ldl -lpthread -lyaml-cpp -lcuda \
# 	$(shell pkg-config --libs $(PKGS))
# $(TEST_BIN): $(TEST_OBJS) align_functions.o svd_rank_cuda.o align_functions_cuda.o elementwise_minus_cuda.o var_axis0_cuda.o matrix_ops_cuda.o matrix_ops_cuda_3x3.o $(INCS) Makefile
# 	$(CXX) -o $@ $(TEST_OBJS) align_functions.o svd_rank_cuda.o align_functions_cuda.o elementwise_minus_cuda.o var_axis0_cuda.o matrix_ops_cuda.o matrix_ops_cuda_3x3.o $(TEST_LIBS)
# tests/%.o: tests/%.cpp $(INCS) Makefile
# 	$(CXX) -c -o $@ $(CFLAGS) $<
# .PHONY: test
# test: $(TEST_BIN)
# 	./$(TEST_BIN) || true

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

# Use NVCCFLAGS without bare -fPIC to avoid nvcc error
NVCCFLAGS := $(filter-out -fPIC -pthread,$(CFLAGS))
%.o: %.cu Makefile
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler -fPIC $<

$(LIB): $(OBJS) $(DEP) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)

