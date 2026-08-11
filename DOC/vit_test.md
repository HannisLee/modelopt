# FP16 ViT ONNX PTQ GPU 测试记录

本文记录 `examples/onnx_ptq` 的一次端到端 GPU 验证。测试使用新建的 Conda 环境
`/root/.conda/envs/modelopt`，并将模型、校准数据及量化结果统一放在仓库根目录的
`model_home/`。

## 测试目标与结果

- 模型：`timm` 的预训练 `vit_base_patch16_224`
- 输入模型：FP16 ONNX，输入形状为 `1x3x224x224`
- 量化：INT8 static PTQ，`max` calibration
- 校准集：Tiny-ImageNet 的 64 张图像，FP16 NPY
- 校准执行提供者：`CUDAExecutionProvider`（GPU 0），CPU 作为回退
- 量化结果：成功；共 155 个节点被量化，产生 240 个 `QuantizeLinear` 和 240 个
  `DequantizeLinear` 节点
- GPU 推理：成功，实际 provider 为 `CUDAExecutionProvider`；输入、输出均为 FP16，
  输出形状为 `1x1000`

本次量化耗时为 26.6 秒。FP16 原始模型为 173,329,840 bytes，INT8 PTQ 模型为
174,105,903 bytes。Q/DQ PTQ 模型的体积不一定会小于 FP16 ONNX，因为量化 scale、
zero point、Q/DQ 图节点和共享常量复制都会增加序列化开销；部署到 TensorRT 后才是
衡量实际引擎体积和性能的合适方式。

## 产物

所有产物位于 [`model_home/`](../model_home/)。

| 文件 | 说明 |
| --- | --- |
| `vit_base_patch16_224.fp16.onnx` | 预训练 ViT 的 FP16 ONNX 模型 |
| `calib.fp16.npy` | 64 张 Tiny-ImageNet 图像构成的 FP16 校准数据，形状为 `(64, 3, 224, 224)` |
| `vit_base_patch16_224.fp16.int8.onnx` | 使用 GPU 校准生成的 INT8 Q/DQ ONNX 模型 |

## 环境与示例依赖

从仓库根目录执行：

```bash
conda activate /root/.conda/envs/modelopt

# 与 torch 2.8.0+cu126 相匹配的 torchvision。
python -m pip install \
    --index-url https://download.pytorch.org/whl/cu126 \
    torchvision==0.23.0

# examples/onnx_ptq/requirements.txt 的依赖。
python -m pip install -r examples/onnx_ptq/requirements.txt
```

本次环境的核心版本为：Python 3.12.13、PyTorch 2.8.0+cu126、CUDA Toolkit 12.6、
`nvidia-modelopt==0.45.0`、`onnxruntime-gpu==1.24.2` 和 `tensorrt-cu12==10.7.0`。

## 完整复现流程

以下命令以仓库根目录为起点。先创建并进入统一的产物目录：

```bash
conda activate /root/.conda/envs/modelopt
mkdir -p model_home
cd model_home
```

### 1. 导出 FP16 ONNX

脚本会下载 `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k` 的公开预训练权重。若直接
访问 Hugging Face 不可用，可像本次测试一样临时使用镜像端点：

```bash
HF_ENDPOINT=https://hf-mirror.com python ../examples/onnx_ptq/download_example_onnx.py \
    --timm_model_name=vit_base_patch16_224 \
    --onnx_save_path=vit_base_patch16_224.fp16.onnx \
    --fp16
```

### 2. 生成 FP16 校准数据

README 建议 CNN/ViT 使用至少 500 张图像。这里使用 64 张图像以验证完整流程；生产或
精度评估应增加 `--calibration_data_size` 到 500 或以上。

```bash
HF_ENDPOINT=https://hf-mirror.com python ../examples/onnx_ptq/image_prep.py \
    --calibration_data_size=64 \
    --model_name=vit_base_patch16_224 \
    --output_path=calib.fp16.npy \
    --fp16
```

### 3. 用 GPU 执行 INT8 PTQ

显式传入 `--calibration_eps cuda:0 cpu`。默认 provider 列表还会包含 TensorRT EP；在本次
FP16 图和 ORT 1.24.2 组合上，该路径会触发 `CopyTensorAsync is not implemented`。CUDA EP
可正常完成校准和量化，CPU 仅作为 ORT 回退 provider。

```bash
python -m modelopt.onnx.quantization \
    --onnx_path=vit_base_patch16_224.fp16.onnx \
    --quantize_mode=int8 \
    --calibration_data_path=calib.fp16.npy \
    --calibration_method=max \
    --calibration_eps cuda:0 cpu \
    --output_path=vit_base_patch16_224.fp16.int8.onnx
```

成功日志中的关键内容如下：

```text
Successfully enabled 2 EPs for ORT:
    [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
Quantization completed successfully in 26.5875 seconds
Total number of quantized nodes: 155
Quantized onnx model is saved as vit_base_patch16_224.fp16.int8.onnx
```

### 4. 验证量化模型可在 GPU 上推理

`onnxruntime-gpu` 的独立推理进程需要先加载 pip 安装的 CUDA/cuDNN 动态库；
`onnxruntime.preload_dlls()` 会完成该步骤。

```bash
python -c "import numpy as np, onnx, onnxruntime as ort; \
x=np.load('calib.fp16.npy', mmap_mode='r')[:1]; \
ort.preload_dlls(); \
m=onnx.load('vit_base_patch16_224.fp16.int8.onnx', load_external_data=False); \
onnx.checker.check_model(m); \
s=ort.InferenceSession('vit_base_patch16_224.fp16.int8.onnx', \
providers=[('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']); \
assert s.get_providers()[0] == 'CUDAExecutionProvider', s.get_providers(); \
y=s.run(None, {s.get_inputs()[0].name: x})[0]; \
print(s.get_providers(), y.shape, y.dtype)"
```

本次输出：

```text
['CUDAExecutionProvider', 'CPUExecutionProvider'] (1, 1000) float16
```

## 未执行项

README 中的 `evaluate.py` 会访问 gated 的 ImageNet-1k 数据集，需要用户提供 Hugging Face
token 和相应访问权限，因此本次未运行。PTQ 导出、ONNX checker 和量化模型的 CUDA 推理均已完成。
