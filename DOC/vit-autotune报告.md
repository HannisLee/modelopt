# ViT AutoTune 测试汇总报告

更新：2026-08-11。本文汇总仓库中已保存的所有 ViT AutoTune/PTQ 系列结果，并加入本次
exact-edge integrated API s50 复测。

## 结论

- 历史 integrated API 的搜索模型可以明显快于其最终 ORT 重建模型；旧 s50 为搜索 P50
  `0.949219 ms`、最终 P50 `1.08545 ms`。
- 本次 exact-edge 路径不再将搜索结果降级为 node-level ORT 配置。新 s50 的搜索模型和最终
  校准模型保持完全相同的 QDQ 节点及 consumer-input edge：均为 `237 Q / 237 DQ / 250 edges`。
- 新 exact-edge s50 的正式 strongly-typed TensorRT 测试中，搜索模型 P50 为 `1.00146 ms`，
  最终校准模型为 `1.05884 ms`。二者仅差真实 scale/per-channel weight 参数，拓扑没有变化。
- 历史结果和本次 exact-edge formal result 的 builder flags 不同（前者常用 `--fp16`，后者使用
  `--stronglyTyped`）；因此跨组仅作趋势参考，search/final 的同组比较才是严格对比。

## 测试配置与数据来源

- GPU：NVIDIA L4，TensorRT 10.7.0。
- 输入：ViT，静态 `1x3x224x224` FP16；校准集为
  `model_home/vit/calib.fp16.npy`（64 张）。
- 历史 formal benchmark：`warmUp=1000`、`iterations=1000`、`noDataTransfers`。
- 历史数据来源：`DOC/08.07.4种autotune汇总报告.md`、
  `DOC/08.10-Autotune报告.md` 和对应 `model_home/vit/*.log`。
- 本次 integrated exact-edge s50：使用 Python API 而不是 CLI，驱动脚本为
  `run_vit_integrated_autotune50_exact.py`；脚本强制当前仓库位于 `sys.path` 首位并断言导入
  `modelopt/onnx/quantization/quantize.py`，不会使用 conda 安装的 0.45 wheel。
- 本次 formal benchmark 使用 `--stronglyTyped`；搜索模型与最终模型使用相同 builder、
  `warmUp=1000`、`iterations=1000`、`noDataTransfers`。

## 全部历史与当前性能结果

P50/P90/P99 均为毫秒。SpeedUp 以历史 FP16 基线 `744.412 QPS` 为分母。

| 系列 | 模型 | s | QPS | SpeedUp | P50 | P90 | P99 | 平均 GPU Compute | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 baseline | `vit.fp16` | - | 744.412 | 1.000× | 1.34351 | 1.35883 | 1.36096 | 1.33869 | PASSED |
| Direct AutoTune | `vit.fp16.autotune15` | 15 | 852.244 | 1.145× | 1.17651 | 1.18481 | 1.18677 | 1.17039 | PASSED |
| Direct AutoTune | `vit.fp16.autotune50` | 50 | 1023.650 | 1.375× | 0.975891 | 0.979004 | 0.979980 | 0.973150 | PASSED |
| Direct AutoTune | `vit.fp16.autotune100` | 100 | 1055.450 | 1.418× | 0.947144 | 0.950317 | 0.951294 | 0.941570 | PASSED |
| INT8 PTQ | `vit.int8` | - | 1019.400 | 1.369× | 0.976807 | 0.994385 | 0.997314 | 0.977940 | PASSED |
| Q/DQ baseline AutoTune | `vit.int8qdq.autotune15` | 15 | 994.169 | 1.336× | 1.009640 | 1.011720 | 1.013790 | 1.002870 | PASSED |
| Q/DQ baseline AutoTune | `vit.int8qdq.autotune50` | 50 | 1017.060 | 1.366× | 0.986084 | 0.988159 | 0.990234 | 0.980230 | PASSED |
| Q/DQ baseline AutoTune | `vit.int8qdq.autotune100` | 100 | 1050.490 | 1.411× | 0.945190 | 0.965698 | 0.968750 | 0.947460 | PASSED |
| Legacy integrated final | `vit.int8.integrated-autotune15` | 15 | 908.271 | 1.220× | 1.096800 | 1.118160 | 1.121220 | 1.097990 | PASSED |
| Legacy integrated final | `vit.int8.integrated-autotune50` | 50 | 915.374 | 1.230× | 1.085450 | 1.106930 | 1.111080 | 1.089440 | PASSED |
| Legacy integrated final | `vit.int8.integrated-autotune100` | 100 | 910.185 | 1.223× | 1.092650 | 1.109990 | 1.115110 | 1.089150 | PASSED |
| Legacy integrated search | `vit.int8.integrated-search-s15` | 15 | 955.454 | 1.284× | 1.042910 | 1.052730 | 1.054810 | 1.039360 | PASSED |
| Legacy integrated search | `vit.int8.integrated-search-s50` | 50 | 1045.540 | 1.405× | 0.949219 | 0.969727 | 0.973877 | 0.953260 | PASSED |
| Legacy integrated search | `vit.int8.integrated-search-s100` | 100 | 1019.710 | 1.370× | 0.971802 | 0.993286 | 0.995361 | 0.974650 | PASSED |
| Integrated + QDQ baseline | `vit.int8.integrated-qdq-autotune15` | 15 | 849.266 | 1.141× | 1.169430 | 1.186770 | 1.230830 | 1.174330 | PASSED |
| Integrated + QDQ baseline | `vit.int8.integrated-qdq-autotune50` | 50 | 937.483 | 1.260× | 1.070070 | 1.076290 | 1.078250 | 1.060070 | PASSED |
| Integrated + QDQ baseline | `vit.int8.integrated-qdq-autotune100` | 100 | 988.491 | 1.328× | 1.010740 | 1.021970 | 1.026120 | 1.008640 | PASSED |
| Exact-edge integrated search (strongly typed) | `integrated-exact-s50/optimized_final.onnx` | 50 | 1001.050 | 1.345× | 1.001460 | 1.010620 | 1.011720 | 0.995956 | PASSED |
| Exact-edge integrated final (strongly typed) | `vit.int8.integrated-exact-autotune50` | 50 | 946.119 | 1.271× | 1.058840 | 1.065920 | 1.067020 | 1.053930 | PASSED |

## 构建产物

历史构建时间、ONNX 大小和 engine 大小已完整记录在
`DOC/08.07.4种autotune汇总报告.md`。本次新增的 exact-edge 产物如下。

| 产物 | ONNX 大小 (MiB) | Engine 大小 (MiB) | Build 状态 | benchmark 日志 |
| --- | ---: | ---: | --- | --- |
| Exact-edge search s50 | 165.39 | 93.55 | PASSED | `model_home/vit/vit.int8.integrated-exact-search-s50.trtexec.log` |
| Exact-edge final s50 | 165.81 | 94.04 | PASSED | `model_home/vit/vit.int8.integrated-exact-autotune50.trtexec.log` |

## QDQ 拓扑对比

| s50 模型 | Q | DQ | initializer Q | activation Q | 带 per-channel axis 的 Q | exact consumer edges |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical integrated search | 195 | 195 | 73 | 122 | 0 | - |
| Historical integrated final | 183 | 183 | 97 | 86 | 49 | - |
| Exact-edge search | 237 | 237 | 77 | 160 | 0 | 250 |
| Exact-edge final | 237 | 237 | 77 | 160 | 36 | 250 |

新路径在 calibration 前后满足：

```text
Q node names 相同
DQ node names 相同
(float source, Q, DQ, consumer, input_index) 集合相同
```

因此，search 到 final 的唯一预期变化是 scale、zero-point、权重 per-channel axis/scale 形状；不会新增、删除或移动 QDQ 边。历史 integrated s50 则发生了 `195 → 183` 的 Q/DQ 减少，且 initializer/activation 的量化语义发生迁移。

## 本次 exact-edge s50 的运行记录

- AutoTune 搜索：624 regions，最多 50 schemes/region；最终 state、pattern cache、候选日志和
  `optimized_final.onnx` 位于 `autotune_output/vit/integrated-exact-s50/`。
- AutoTune Python benchmark（strongly typed，warmup 10、timing 50）：
  - 搜索模型 median：`0.934 ms`。
  - 最终 calibrated 模型 median：`0.971 ms`。
- Formal `trtexec` benchmark（1000 warmup + 1000 iterations）结果见上表；该组中 final 比 search
  慢约 `5.73%`，但拓扑严格相同，差异来自真实 calibration scale 和 per-channel weight 参数。

## 当前判断与后续建议

exact-edge handoff 已解决“搜索 QDQ 被 ORT node-level 重建”的结构性问题：新 final 的边界与搜索结果一致。
但 final 仍比 search 慢，说明真实 non-uniform scale 对 TensorRT fusion/requantization 仍有影响；这一点与
`DOC/08.11-Autotune审计报告.md` 中的 scale-mismatch 判断一致。

下一步应在 exact topology 固定的前提下，对 top-K scheme 用真实 scale 重校准、重建并排序；这样才能将
AutoTune 的选择目标从占位 `0.1` scale 图转为最终可部署图。
