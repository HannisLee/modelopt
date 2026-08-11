# ViT ONNX AutoTune 校准流程

本文约定 ViT 的最终模型、评测日志和 TensorRT engine 均在 `model_home/vit/`，候选搜索输出均在 `autotune_output/vit/`。

## 输入与命名

- 原始 FP16：`model_home/vit/vit.fp16.onnx`
- INT8 Q/DQ baseline：`model_home/vit/vit.int8.onnx`
- 校准数据：`model_home/vit/calib.fp16.npy`
- `s<数量>` 表示 `schemes_per_region`，例如 `s50` 为每个 region 搜索 50 个候选方案。

## 样例命令

以下示例将 direct FP16 AutoTune-50 放到 GPU 0；搜索过程日志单独保存。

```bash
mkdir -p model_home/vit/autotune-logs/direct

CUDA_VISIBLE_DEVICES=0 python -m modelopt.onnx.quantization.autotune \
  --onnx_path ./model_home/vit/vit.fp16.onnx \
  --output_dir ./autotune_output/vit/direct-fp16-s50 \
  --timing_cache ./autotune_output/vit/direct-fp16-s50/trt_timing.cache \
  --quant_type int8 \
  --schemes_per_region 50 \
  --warmup_runs 10 \
  --timing_runs 50 \
  > ./model_home/vit/autotune-logs/direct/direct-fp16-s50.log 2>&1
```

Q/DQ baseline AutoTune-50 的差别是增加 `--qdq_baseline`，并使用 `qda-s50` 目录：

```bash
mkdir -p model_home/vit/autotune-logs/qda

CUDA_VISIBLE_DEVICES=0 python -m modelopt.onnx.quantization.autotune \
  --onnx_path ./model_home/vit/vit.fp16.onnx \
  --qdq_baseline ./model_home/vit/vit.int8.onnx \
  --output_dir ./autotune_output/vit/qda-s50 \
  --timing_cache ./autotune_output/vit/qda-s50/trt_timing.cache \
  --quant_type int8 \
  --schemes_per_region 50 \
  --warmup_runs 10 \
  --timing_runs 50 \
  > ./model_home/vit/autotune-logs/qda/qda-s50.log 2>&1
```

Integrated API AutoTune-50 同时完成 INT8 校准和候选搜索。当前环境使用 `--calibration_eps cpu` 完成最终静态校准，以避开 CUDA/TensorRT EP 的 `CopyTensorAsync` 限制；它不影响后续 `trtexec` 部署评测。

```bash
mkdir -p model_home/vit/autotune-logs/integrated

CUDA_VISIBLE_DEVICES=0 python -m modelopt.onnx.quantization \
  --onnx_path ./model_home/vit/vit.fp16.onnx \
  --quantize_mode int8 \
  --calibration_data_path ./model_home/vit/calib.fp16.npy \
  --output_path ./model_home/vit/vit.int8.integrated-autotune50.onnx \
  --autotune \
  --autotune_output_dir ./autotune_output/vit/integrated-s50 \
  --autotune_schemes_per_region 50 \
  --autotune_warmup_runs 10 \
  --autotune_timing_runs 50 \
  --calibration_eps cpu \
  > ./model_home/vit/autotune-logs/integrated/integrated-s50.log 2>&1
```

将示例中的 `50` 同步替换为 `15` 或 `100`，即可得到相应 schemes 数量的任务。并发运行时，每个 integrated 任务应使用独立的 FP16 输入副本，避免预处理阶段同时修改同一个输入文件。

## 输出目录规范

| 方法 | 输出目录 |
| --- | --- |
| direct FP16 | `autotune_output/vit/direct-fp16-s<数量>/` |
| Q/DQ baseline | `autotune_output/vit/qda-s<数量>/` |
| integrated API | `autotune_output/vit/integrated-s<数量>/` |

每个输出目录保留 `autotuner_state.yaml`、pattern cache、`trt_timing.cache`（如有）、候选 region 模型和 `optimized_final.onnx`。重跑同一个目录时会使用 state 续跑；不要在开始新实验前删除该目录。

`autotune_output/vit/direct-legacy-s15/` 是历史 15-schemes direct 结果，仅用于保留旧实验，不作为新任务命名模板。

## 最终文件与日志位置

- 最终 ONNX：`model_home/vit/<模型名>.onnx`
- TensorRT engine：`model_home/vit/plan/<模型名>.plan`
- `trtexec` 构建与 1,000 次评测日志：`model_home/vit/<模型名>.log`
- AutoTune 搜索原始日志：`model_home/vit/autotune-logs/<方法>/<任务名>.log`

例如 direct 50 的最终模型、engine 和评测日志分别为：

```text
model_home/vit/vit.fp16.autotune50.onnx
model_home/vit/plan/vit.fp16.autotune50.plan
model_home/vit/vit.fp16.autotune50.log
```

## ResNet50 固定 shape INT8 Q/DQ warm-start

ResNet50 实验使用静态 batch=1，模型和实验输出分别位于 `model_home/resnet50/` 与 `autotune_output/resnet50/`。原始模型的输入名是 `data`，固定 shape 时不能使用 ViT 示例中的输入名：

```bash
polygraphy surgeon sanitize \
  --override-input-shapes data:[1,3,224,224] \
  -o model_home/resnet50/resnet50_bs1.onnx \
  model_home/resnet50/resnet50.onnx
```

标准 INT8 PTQ 先在独立输入副本上运行，生成 `resnet50.int8.onnx` 作为后续 Q/DQ warm-start。ResNet 的 `data` 输入为 FP32，因此校准数组也必须是 `[N,3,224,224]` FP32；若复用 ViT 的 FP16 数组，应先转换为 `model_home/resnet50/calib.fp32.npy`。这只保证 shape/dtype 兼容，并不等同于 ResNet 专用预处理。

```bash
CUDA_VISIBLE_DEVICES=3 /root/.conda/envs/modelopt/bin/python -m modelopt.onnx.quantization \
  --onnx_path ./model_home/resnet50/inputs/resnet50.ptq.onnx \
  --quantize_mode int8 \
  --calibration_data_path ./model_home/resnet50/calib.fp32.npy \
  --calibration_eps cpu \
  --output_path ./model_home/resnet50/resnet50.int8.onnx
```

PTQ 通过 `onnx.checker` 后，使用 GPU 0/1/2 并行运行 integrated AutoTune 的 s30、s50、s100。每个任务必须拥有独立的 `resnet50.s<数量>.onnx` 输入副本，以及独立的 `integrated-qdq-s<数量>/` 输出目录、state 和 timing cache：

```bash
CUDA_VISIBLE_DEVICES=<gpu> /root/.conda/envs/modelopt/bin/python -m modelopt.onnx.quantization \
  --onnx_path ./model_home/resnet50/inputs/resnet50.s<s>.onnx \
  --quantize_mode int8 \
  --calibration_data_path ./model_home/resnet50/calib.fp32.npy \
  --calibration_eps cpu \
  --output_path ./model_home/resnet50/resnet50.int8.integrated-qdq-autotune<s>.onnx \
  --autotune \
  --autotune_output_dir ./autotune_output/resnet50/integrated-qdq-s<s> \
  --autotune_state_file ./autotune_output/resnet50/integrated-qdq-s<s>/autotuner_state.yaml \
  --autotune_timing_cache ./autotune_output/resnet50/integrated-qdq-s<s>/trt_timing.cache \
  --autotune_qdq_baseline ./model_home/resnet50/resnet50.int8.onnx \
  --autotune_schemes_per_region <s> \
  --autotune_warmup_runs 50 \
  --autotune_timing_runs 100
```

这里 `<s>` 为 30、50、100。每个搜索目录保留 `autotuner_state.yaml`、pattern cache、`trt_timing.cache`、region artifacts 与 `optimized_final.onnx`；搜索最佳 ONNX 和 integrated 最终静态校准 ONNX 都须通过 `onnx.checker`。
