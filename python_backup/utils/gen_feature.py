import os
import configparser
import toml
import math
import cv2
import tensorrt as trt
import numpy as np
# Compatibility for NumPy>=1.24 where np.bool alias was removed; TensorRT<=8.x uses np.bool in nptype()
if not hasattr(np, "bool"):
    np.bool = np.bool_
from cuda import cuda, cudart

def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res

# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))

# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))



class TensorRTInfer:
    """
    Implements inference for TensorRT engine.
    """

    def __init__(self, engine_path, mode='max'):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        :param max: infer mode, 'max' or 'min', batch size
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            shape = self.context.get_tensor_shape(name)
            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_tensor_profile_shape(name, 0)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                if mode == 'max':
                    self.context.set_input_shape(name, profile_shape[2])
                elif mode == 'min':
                    self.context.set_input_shape(name, profile_shape[0])
                shape = self.context.get_tensor_shape(name)
            if is_input:
                self.batch_size = shape[0]
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = cuda_call(cudart.cudaMalloc(size))
            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            print(
                "{} '{}' with shape {} and dtype {}".format(
                    "Input" if is_input else "Output",
                    binding["name"],
                    binding["shape"],
                    binding["dtype"],
                )
            )

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def infer(self, batch):
        """
        Execute inference on a batch of images.
        :param batch: A numpy array holding the image batch.
        :return A list of outputs as numpy arrays.
        """
        # Copy I/O and Execute
        memcpy_host_to_device(self.inputs[0]["allocation"], batch)
        self.context.execute_v2(self.allocations)
        for o in range(len(self.outputs)):
            memcpy_device_to_host(
                self.outputs[o]["host_allocation"], self.outputs[o]["allocation"]
            )
        return [o["host_allocation"] for o in self.outputs]
    
def preprocess(input_path, netshape):
    raw_image = cv2.imread(input_path)
    if raw_image is None:
        raise FileNotFoundError(f"Failed to read image: {input_path}")
    assert raw_image.shape[0] == 112 and raw_image.shape[1] == 112
    rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    # ArcFace normalization: (img - 127.5) / 128.0, CHW
    image_data = rgb.astype(np.float32)
    image_data -= 127.5
    image_data /= 128.0
    image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    image = np.array(image_data, dtype=np.float32, order="C")
    return image, raw_image

if __name__ == "__main__":
    # Resolve SGIE engine path from the pipeline TOML -> SGIE config -> model-engine-file
    engine_path = None
    try:
        cfg = toml.load("config/config_pipeline.toml")
        sgie_cfg_path = cfg.get('sgie', {}).get('config-file-path', '').strip()
        if sgie_cfg_path and os.path.exists(sgie_cfg_path):
            cp = configparser.ConfigParser()
            cp.read(sgie_cfg_path)
            if cp.has_option('property', 'model-engine-file'):
                engine_path = cp.get('property', 'model-engine-file')
        # Resolve relative path to workspace root if needed
        if engine_path and not os.path.isabs(engine_path):
            engine_path = os.path.normpath(os.path.join(os.path.dirname(sgie_cfg_path), engine_path))
    except Exception:
        engine_path = None
    # Fallbacks
    if not engine_path or not os.path.exists(engine_path):
        cand = "./models/arcface/glintr100.onnx_b4_gpu0_fp16.engine"
        engine_path = cand if os.path.exists(cand) else "./models/arcface/arcface.engine"
    model = TensorRTInfer(engine_path, mode='min')
    _shape, _dtype = model.input_spec()
    path = "data/known_faces"
    # Only process images; skip existing .npy files
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    faces = [f for f in os.listdir(path) if os.path.splitext(f)[1].lower() in exts]
    if not faces:
        print(f"No images found in {path}; expected one of {sorted(list(exts))}")
    for face in faces:
        image_path = os.path.join(path, face)
        try:
            # Preprocess with RGB + ArcFace normalization
            raw = cv2.imread(image_path)
            if raw is None:
                print(f"Skip unreadable file: {image_path}")
                continue
            if raw.shape[0] != 112 or raw.shape[1] != 112:
                print(f"Skip non-112x112 image: {image_path} (got {raw.shape[1]}x{raw.shape[0]})")
                continue
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            img = rgb.astype(np.float32)
            img -= 127.5
            img /= 128.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0).astype(np.float32)
            inp = np.array(img, dtype=np.float32, order="C")

            preds = model.infer(inp)[0]
            res = np.reshape(preds, (1, -1))
            # L2 normalize
            norm = np.linalg.norm(res)
            if norm == 0:
                print(f"Zero norm embedding for {image_path}; skipping save")
                continue
            normal_array = (res / norm).reshape(-1, 1)

            out_npy_path = os.path.join(path, os.path.splitext(face)[0] + ".npy")
            np.save(out_npy_path, normal_array)
            print(f"Saved embedding: {out_npy_path} shape={normal_array.shape}")
        except Exception as e:
            print(f"Failed on {image_path}: {e}")

