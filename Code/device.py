
import tensorrt as trt
import torch

print("TensorRT:", trt.__version__)
print("Torch:", torch.__version__)
print("Built CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

import onnxruntime as ort

print("ONNX Runtime:", ort.__version__)
print("Available providers:", ort.get_available_providers())