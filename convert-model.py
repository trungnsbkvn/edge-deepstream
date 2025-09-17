'''
Author: zhouyuchong
Date: 2024-05-30 11:15:58
Description: Convert ONNX to TensorRT engine (multi-input supported, auto-detect input size, precision modes)
LastEditors: ChatGPT
LastEditTime: 2025-09-08
'''
import os
import argparse
from loguru import logger
import tensorrt as trt
import math
import onnx


TRT_LOGGER = trt.Logger()


def detect_input_size(onnx_file_path):
    """Read ONNX model and detect input spatial size (H, W)."""
    model = onnx.load(onnx_file_path)
    for inp in model.graph.input:
        shape = [d.dim_value if (d.dim_value > 0) else -1 for d in inp.type.tensor_type.shape.dim]
        # Look for image-like input (NCHW)
        if len(shape) == 4 and (shape[1] in [1, 3]):  # likely image tensor
            h, w = shape[2], shape[3]
            return h, w
    return None, None


def build_engine(onnx_file_path, engine_file_path="", set_input_shape=None, precision="fp32"):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(network_creation_flag) as network, \
         builder.create_builder_config() as config, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         trt.Runtime(TRT_LOGGER) as runtime:

        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, 1 << 28
        )  # 256MiB

        # Precision flags
        if precision == "fp16":
            if not builder.platform_has_fast_fp16:
                logger.warning("FP16 not supported on this device, falling back to FP32")
            else:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("Using FP16 mode")
        elif precision == "int8":
            if not builder.platform_has_fast_int8:
                logger.warning("INT8 not supported on this device, falling back to FP32")
            else:
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("Using INT8 mode (calibration required!)")

        if not os.path.exists(onnx_file_path):
            raise FileNotFoundError(f"ONNX file {onnx_file_path} not found.")

        print(f"Loading ONNX file from path {onnx_file_path}...")
        with open(onnx_file_path, "rb") as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error_idx in range(parser.num_errors):
                    print(parser.get_error(error_idx))
                return None

        # Create optimization profile
        profile = builder.create_optimization_profile()

        logger.debug(f"total input layer: {network.num_inputs}")
        logger.debug(f"total output layer: {network.num_outputs}")

        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            logger.debug(f"input layer-{i}: {input_tensor.name} | shape: {input_tensor.shape}")

            # If input is image-like (4D tensor: NCHW)
            if len(input_tensor.shape) == 4:
                profile.set_shape(
                    input_tensor.name,
                    set_input_shape[0],  # min
                    set_input_shape[1],  # opt
                    set_input_shape[2]   # max
                )
            else:
                # Keep static shape (usually im_info is (1,2) or (1,3))
                static_shape = tuple(dim if dim > 0 else 1 for dim in input_tensor.shape)
                profile.set_shape(input_tensor.name, static_shape, static_shape, static_shape)
                logger.debug(f"Static shape profile applied for {input_tensor.name}")

        config.add_optimization_profile(profile)

        logger.debug("build, may take a while...")

        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError("Failed to build engine. Check ONNX and input shapes.")

        engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")

        with open(engine_file_path, "wb") as f:
            f.write(plan)

        return engine


def main(args):
    # Auto-detect input size if not provided
    if args.size == "auto":
        h, w = detect_input_size(args.onnx)
        if h is None or w is None:
            raise RuntimeError("Failed to auto-detect input size, please specify with -s HxW")
        logger.info(f"Auto-detected input size: {h}x{w}")
    else:
        size = [int(x) for x in args.size.split('x')]
        h, w = size[0], size[1]

    # Generate descriptive engine file name
    base_name = os.path.splitext(args.onnx)[0]
    trt_file_name = f"{base_name}_{h}x{w}_bs{args.batch}_{args.precision}.trt"

    input_shape = [
        (1, 3, h, w),
        (math.ceil(args.batch / 2), 3, h, w),
        (args.batch, 3, h, w)
    ]
    logger.debug(f"set input shape: {input_shape}")
    build_engine(args.onnx, trt_file_name, input_shape, args.precision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, default='', help='onnx path', required=True)
    parser.add_argument('-s', '--size', type=str, default="auto", help='input shape, e.g. 640x640, or \"auto\"')
    parser.add_argument('-b', '--batch', type=int, default=1, help='max batch size')
    parser.add_argument('--precision', type=str, choices=["fp32", "fp16", "int8"], default="fp32", help='precision mode')
    args = parser.parse_args()
    main(args=args)
