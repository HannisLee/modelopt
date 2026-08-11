# ONNX Autotune 执行流程与核心逻辑

本文梳理以下命令在 ModelOpt 中的执行入口、文件调用顺序、核心数据结构和搜索逻辑：

```bash
python -m modelopt.onnx.quantization.autotune --onnx_path model.onnx
```

这条命令的本质是：把 ONNX 计算图划分成若干 region，将结构相同的 region 归为同一种 pattern；针对每种 pattern 生成多种 Q/DQ 插入方案，逐个构建完整 TensorRT Engine 并实测延迟，最后选择延迟最低的方案导出模型。

它优化的是 **Q/DQ 放置位置**，不是 calibration scale，也不评估模型精度。

## 1. 文件进入顺序

使用 `python -m` 执行一个 package 时，Python 会先初始化各层 package，然后执行目标 package 的 `__main__.py`：

```text
modelopt/__init__.py
  ↓
modelopt/onnx/__init__.py
  ↓
modelopt/onnx/quantization/__init__.py
  ↓
modelopt/onnx/quantization/autotune/__init__.py
  ↓
modelopt/onnx/quantization/autotune/__main__.py
```

各文件的职责如下：

1. `modelopt/__init__.py`
   - 初始化 `modelopt` package。
   - 读取 `nvidia-modelopt` 包版本。

2. `modelopt/onnx/__init__.py`
   - 导入 ONNX quantization 子包。
   - 检查 ONNX 可选依赖和 Python 版本。

3. `modelopt/onnx/quantization/__init__.py`
   - 暴露高层 `quantize()` 和 `quantize_int4()` API。

4. `modelopt/onnx/quantization/autotune/__init__.py`
   - 导入 `QDQAutotuner`、Benchmark、Region、PatternCache 等 Autotune 核心类型。

5. `modelopt/onnx/quantization/autotune/__main__.py`
   - Autotune CLI 的实际入口。

入口代码为：

```python
if __name__ == "__main__":
    sys.exit(run_autotune())
```

业务层面的主要调用链为：

```text
run_autotune()
  ├─ get_parser()
  ├─ apply_mode_presets()
  ├─ validate_file_path()
  ├─ init_benchmark_instance()
  └─ region_pattern_autotuning_workflow()
       ├─ onnx.load()
       ├─ QDQAutotuner(...)
       ├─ autotuner.initialize()
       │    └─ CombinedRegionSearch.search_regions()
       ├─ 测量 baseline
       ├─ region/pattern profiling 循环
       │    ├─ set_profile_region()
       │    ├─ generate()
       │    ├─ export_onnx()
       │    ├─ benchmark_onnx_model()
       │    └─ submit()
       └─ 导出 optimized_final.onnx
```

## 2. CLI 入口与默认参数

CLI 核心函数是 `modelopt/onnx/quantization/autotune/__main__.py` 中的 `run_autotune()`。

对于最简命令：

```bash
python -m modelopt.onnx.quantization.autotune --onnx_path model.onnx
```

实际采用的主要默认参数是：

```text
onnx_path        = model.onnx
output_dir       = ./autotuner_output
mode             = default
schemes/region   = 50
warmup_runs      = 50
timing_runs      = 100
quant_type       = int8
default_dq_dtype = float32
use_trtexec      = False
timing_cache     = /tmp/trtexec_timing.cache
```

默认值主要定义在：

- `modelopt/onnx/quantization/autotune/__main__.py`
- `modelopt/onnx/quantization/autotune/utils.py`

`run_autotune()` 的执行顺序如下：

1. `get_parser().parse_args()` 解析命令行参数。
2. `apply_mode_presets()` 应用 `quick/default/extensive` 模式预设。
3. `validate_file_path()` 检查 ONNX 模型等输入文件。
4. `init_benchmark_instance()` 初始化 TensorRT Benchmark。
5. `get_node_filter_list()` 加载可选节点过滤规则。
6. 调用 `region_pattern_autotuning_workflow()` 执行完整搜索。

由于默认没有指定 `--use_trtexec`，Benchmark 默认走：

```text
TensorRTPyBenchmark
```

而不是 `TrtExecBenchmark`。

## 3. 高层工作流

完整流程由以下函数控制：

```python
region_pattern_autotuning_workflow(...)
```

它位于：

```text
modelopt/onnx/quantization/autotune/workflows.py
```

### 3.1 创建输出目录

默认创建：

```text
autotuner_output/
├── logs/
└── region_models/
```

如果没有显式指定 `--state_file`，状态文件默认为：

```text
autotuner_output/autotuner_state.yaml
```

### 3.2 加载模型

```python
model = onnx.load(model_or_path)
```

然后创建并初始化 Autotuner：

```python
autotuner = QDQAutotuner(model)
autotuner.initialize(config, pattern_cache)
```

如果状态文件已经存在，则调用：

```python
autotuner.load_state(state_file)
```

尝试恢复以前已经完成的搜索结果。

如果指定了 `--qdq_baseline`，还会从现有 QDQ 模型中提取已量化 tensor，并将其转换为 pattern cache 的初始候选方案。

## 4. Region 是怎么发现的

`QDQAutotuner.initialize()` 首先调用基类初始化，然后执行：

```python
self._search_regions()
```

实际使用的 region 搜索器是：

```python
CombinedRegionSearch(...)
```

它位于：

```text
modelopt/onnx/quantization/autotune/region_search.py
```

Region 搜索分为两个阶段。

### 4.1 阶段一：Bottom-up partitioning

`RegionPartitioner.partition_graph()` 按 ONNX 图中的节点顺序遍历计算图，主要识别三类结构。

#### 线性序列

```text
A → B → C → D
```

连续的非分叉节点会被组合成一个 LEAF region。

#### 分叉后汇合

```text
       B
     ↗   ↘
A           D
     ↘   ↗
       C
```

如果一个节点的输出分叉，且这些分支在有限距离内重新汇合，搜索器会尝试将分叉点、分支节点和汇合点组织在同一个 region 中。

#### 分叉但不汇合

```text
A → B
  ↘ C
```

如果没有找到合理的汇合点，分叉节点可能被单独放入一个 region，后继节点再分别处理。

### 4.2 阶段二：Top-down refinement

对于节点数不少于默认阈值 10 的较大 region，会使用 `TopDownRegionBuilder` 继续细分：

- 初始拆成较小的 LEAF region。
- 根据 producer-consumer 关系重新合并线性序列。
- 识别分叉和汇合子结构。
- 创建层次化 COMPOSITE region。
- 默认限制单个 sequence region 最多包含 10 个节点。

### 4.3 Region 排序

Region 搜索完成后，层次结构会被展平，并按照以下顺序进行 profiling：

```text
LEAF region
  ↓
COMPOSITE region
  ↓
ROOT region
```

也就是说，先优化局部细粒度结构，再处理较高层的组合结构。

## 5. Pattern 是什么

每个 region 都会转换成一个 `RegionPattern`：

```python
RegionPattern.from_region(region, graph)
```

Pattern signature 会编码：

- region 内节点的 op 类型。
- Conv 等算子的关键 attributes。
- Add、Mul 等交换操作的输入结构。
- COMPOSITE region 的子 region 层次关系。
- 节点和子 region 的确定性排列顺序。

例如，一个 pattern 可能类似：

```text
Conv[kernel_shape=3x3]->Relu
```

或者：

```text
COMPOSITE(MatMul->Add|Relu+MatMul->Add)
```

结构相同的 region 会产生相同的 signature。

因此，Autotune 的核心优化单位实际上是 **unique pattern**，而不是每个 region。某个 pattern 已经完成 profiling 后，后续结构相同的 region 会直接复用该 pattern 的最佳 scheme。

## 6. Baseline 测量

正式搜索之前，工作流先导出一个不插入任何 Q/DQ 的模型：

```python
autotuner.export_onnx(
    "autotuner_output/baseline.onnx",
    insert_qdq=False,
)
```

然后调用 TensorRT Benchmark：

```python
baseline_latency = benchmark_onnx_model(...)
autotuner.submit(baseline_latency)
```

第一次调用 `submit()` 时，`baseline_latency_ms` 还是 `None`，因此该延迟只会被记录成整个搜索过程的 baseline。

## 7. 每个 Region/Pattern 如何搜索

工作流的核心循环可以简化为：

```python
for region in regions:
    autotuner.set_profile_region(region)

    for _ in range(num_schemes_per_region):
        scheme_idx = autotuner.generate()
        model_bytes = autotuner.export_onnx(None, insert_qdq=True)
        latency = benchmark_onnx_model(model_bytes)
        autotuner.submit(latency)
```

### 7.1 `set_profile_region()`

这个方法负责：

1. 提交上一个 pattern 的搜索结果。
2. 根据当前 region 创建 `RegionPattern`。
3. 检查相同 pattern 是否已经完成 profiling。
4. 如果存在 pattern cache，加载缓存中的候选方案。
5. 否则创建一个新的空 `PatternSchemes`。

### 7.2 候选插入点

Autotuner 将候选 Q/DQ 插入位置分为三类。

#### `NodeInputInsertionPoint`

表示某个 region 内节点的某一个输入位置。

#### `ChildRegionInputInsertionPoint`

表示 COMPOSITE region 中某个子 region 的输入边界。

#### `ChildRegionOutputInsertionPoint`

表示 region 或子 region 的输出边界。

这些 insertion point 使用的是 **pattern-relative 索引**，而不是整个 ONNX 图中的绝对节点编号。因此同一个 scheme 可以复用到所有结构相同的 region。

候选点收集过程中会过滤不适合量化的位置，例如：

- bool、comparison、shape 类计算。
- Cast、Shape、Identity、Pad 等结构操作。
- 非浮点 tensor。
- 很小的 tensor。
- bias、BatchNormalization 参数等不适合单独量化的输入。
- 某些会破坏 TensorRT fusion 的位置，例如部分 Conv→Relu 中间位置。

对于 Conv、ConvTranspose、Gemm、MatMul 等线性算子，activation 输入和 weight 会成对量化。

## 8. Scheme 如何产生

Scheme 生成的核心方法是：

```python
QDQAutotunerBase.generate()
```

执行逻辑如下：

```text
存在缓存且尚未重新测量的 scheme
  → 优先返回缓存 scheme

否则
  → 从当前表现最好的若干 scheme 中随机选择一个
  → 随机增加、删除或同时增删 Q/DQ 插入点
  → 检查 scheme hash，确保新方案不重复
  → 最多尝试 100 次
```

### 8.1 冷启动时的第一个 Scheme

如果没有缓存，也没有已经测量过的 scheme，第一次生成的是空 scheme：

```text
当前 pattern 不增加任何 Q/DQ
```

该空 scheme 测量完成后，后续候选才会从已经测量的较优 scheme 出发进行随机变异。

### 8.2 随机变异策略

变异会分别作用于：

- node input insertion points；
- child region input insertion points；
- region output insertion points。

每一类会随机选择：

```text
add
remove
both
```

默认每类最多变异 3 个插入点。

当前实现没有在 CLI 工作流中设置固定随机种子，因此不同运行可能得到不同的 scheme 搜索顺序和最终结果。

## 9. Scheme 如何转换成 ONNX 模型

`export_onnx()` 会先把 pattern-relative insertion point 转换成真实 ONNX 图中的位置：

```python
ResolvedInsertionPoint(
    tensor_name=...,
    node_index=...,
    input_index=...,
)
```

主要调用链为：

```text
QDQAutotunerBase.export_onnx()
  ↓
get_resolved_insertion_points()
  ↓
RegionPattern.matches(..., scheme)
  ↓
export_qdq_onnx()
  ↓
insert_qdq_at_tensors()
  ↓
create_qdq_nodes()
```

Q/DQ 插入后的结构为：

```text
原 tensor
   ↓
QuantizeLinear
   ↓
DequantizeLinear
   ↓
原 consumer node
```

默认 INT8 Q/DQ 参数为：

```text
scale      = 0.1
zero_point = 0
quant type = int8
DQ dtype   = float32
```

如果选择 FP8，当前实现会先构造 INT8 Q/DQ 模型，再调用 `int8_to_fp8()` 转换成 FP8。

### 9.1 层次化 Region 的覆盖关系

Resolved insertion points 按 region 顺序合并。由于 LEAF region 先处理，COMPOSITE region 后处理，高层 region 在应用自己的 scheme 时会先移除与该 region 重叠的已有插入点，再加入自己的插入点。

因此，高层 COMPOSITE scheme 可以覆盖局部 LEAF scheme 中重叠的 Q/DQ 决策。

## 10. 每个 Scheme 怎么测试

每个候选 scheme 不是只测试当前 region，而是导出一个完整模型，其中包含：

```text
之前已经提交的 pattern 最佳方案
  +
当前 pattern 的候选方案
```

随后重新构建整个 TensorRT Engine 并测量整体延迟。

默认 `TensorRTPyBenchmark` 流程为：

```text
解析 ONNX
  ↓
创建 strongly-typed TensorRT network
  ↓
应用 timing cache
  ↓
构建 serialized TensorRT Engine
  ↓
反序列化 Engine
  ↓
创建 ExecutionContext
  ↓
分配 host/device buffer
  ↓
执行 50 次 warmup
  ↓
执行 100 次 timing
  ↓
取 median latency
```

### 10.1 计时范围

正式 timing 时，计时区间只覆盖：

```python
context.execute_async_v3(...)
```

输入 H2D 和输出 D2H 拷贝不包含在最终 latency 内。

### 10.2 输入数据和动态 Shape

Benchmark 默认：

- 使用随机正态分布生成输入数据。
- 将动态维度 `-1` 替换成 `1`。
- CLI 当前没有直接暴露 `TensorRTPyBenchmark.set_shapes()`。

因此，对于动态 shape 模型，默认测到的可能是所有动态维度都取 1 的延迟，而不是生产环境中的真实 shape。

### 10.3 Benchmark 失败

如果 ONNX 解析、Engine 构建、反序列化或执行失败，Benchmark 返回：

```python
float("inf")
```

对应 scheme 会被标记为 `error=True`，并从最佳方案选择中排除。

Timing cache 默认保存在：

```text
/tmp/trtexec_timing.cache
```

工作流每 10 个候选要求 Benchmark 将 timing cache 刷新到磁盘。

## 11. 最佳方案如何确定

`submit(latency)` 会把延迟写入当前 `InsertionScheme`，然后按照 latency 排序。

`PatternSchemes.best_scheme` 的逻辑是：

```text
排除 error=True 的 scheme
  ↓
选择 latency_ms 最小的 scheme
```

当前代码中的 `Config.performance_threshold=1.02` 没有被实际用于最佳方案选择。因此：

- 没有强制要求候选必须比 baseline 快 2%。
- 最终选择的是所有有效候选中延迟最低的方案。
- 因为候选包含空 scheme，某些 pattern 最终可能选择不插入 Q/DQ。

## 12. Pattern Cache

每次 `submit()` 后，当前 pattern 的 scheme 会被加入 `PatternCache`。

Cache 会：

- 删除失败 scheme。
- 按 scheme hash 去重。
- 对距离过近的相似 scheme 只保留延迟更低的一个。
- 每个 pattern 默认最多保留 32 个 scheme。
- 默认要求被保留 scheme 之间的 edit distance 不小于 4。

下一次对相同或相似模型运行时，可以使用：

```bash
--pattern_cache previous_pattern_cache.yaml
```

缓存 scheme 会被优先重新测量，然后搜索器再从这些候选出发继续变异。

## 13. 状态保存与恢复

每轮 region profiling 后，工作流调用：

```python
autotuner.save_state(state_path)
```

输出：

```text
autotuner_state.yaml
autotuner_state_pattern_cache.yaml
```

需要注意当前实现的一个细节：

- 当前 pattern 通常在切换到下一个 region 时才被正式 commit 到 `profiled_patterns`。
- `autotuner_state.yaml` 主要保存已经 commit 的 pattern。
- 当前 pattern 的候选会进入 pattern cache，但恢复时缓存候选的 latency 会被重置并重新测量。
- 因此异常中断后，最后一个正在处理或刚处理完但尚未 commit 的 pattern 可能需要重新 profiling。

## 14. 最终导出

全部 region 处理完成后，工作流提交最后一个 pattern：

```python
autotuner.set_profile_region(None, commit=True)
```

然后导出最终模型：

```python
autotuner.export_onnx(
    "autotuner_output/optimized_final.onnx",
    insert_qdq=True,
)
```

最后还会重新构建一次 TensorRT Engine，测量最终整体延迟，并与 baseline 比较。

默认输出目录结构如下：

```text
autotuner_output/
├── baseline.onnx
├── optimized_final.onnx
├── autotuner_state.yaml
├── autotuner_state_pattern_cache.yaml
├── logs/
│   ├── baseline.log
│   ├── region_*_scheme_*.log
│   └── final.log
└── region_models/
    └── region_*_level_*.onnx
```

## 15. 完整流程总结

```text
ONNX 模型
  ↓
解析计算图
  ↓
Bottom-up region 划分
  ↓
Top-down 层次化 refinement
  ↓
为每个 region 生成结构签名
  ↓
相同 pattern 去重
  ↓
测量无 Q/DQ baseline
  ↓
为每种 unique pattern 生成 Q/DQ placement scheme
  ↓
导出完整候选 ONNX
  ↓
构建完整 TensorRT Engine
  ↓
实测 median latency
  ↓
选择每个 pattern 的最低延迟 scheme
  ↓
合并所有最佳 scheme
  ↓
导出 optimized_final.onnx
```

## 16. 这条命令不做什么

直接运行 `modelopt.onnx.quantization.autotune` 不负责：

- 使用代表性数据执行 calibration。
- 根据真实数据分布计算 activation scale。
- 精度评估。
- latency/accuracy 多目标权衡。
- 保证输出模型满足精度要求。

因此，如果目标是生产级的“校准 + 量化 + Autotune”，应该优先使用完整 ONNX quantization CLI：

```bash
python -m modelopt.onnx.quantization \
    --onnx_path model.onnx \
    ...校准数据参数... \
    --autotune=default
```

直接运行：

```bash
python -m modelopt.onnx.quantization.autotune --onnx_path model.onnx
```

更适合作为底层 Q/DQ placement 搜索器、算法开发入口和调试工具。

## 17. 推荐代码阅读顺序

如果要继续深入源码，建议按以下顺序阅读：

1. `modelopt/onnx/quantization/autotune/__main__.py`
   - CLI 入口、参数和默认配置。

2. `modelopt/onnx/quantization/autotune/workflows.py`
   - 完整工作流和 profiling 主循环。

3. `modelopt/onnx/quantization/autotune/autotuner.py`
   - `QDQAutotuner` 和自动 region discovery。

4. `modelopt/onnx/quantization/autotune/autotuner_base.py`
   - `set_profile_region()`、`generate()`、`submit()`、`export_onnx()` 等核心状态机。

5. `modelopt/onnx/quantization/autotune/region_search.py`
   - Bottom-up 和 top-down region 划分算法。

6. `modelopt/onnx/quantization/autotune/region_pattern.py`
   - Region pattern signature 和 pattern-relative scheme。

7. `modelopt/onnx/quantization/autotune/insertion_points.py`
   - 候选 Q/DQ 插入点收集、过滤和解析。

8. `modelopt/onnx/quantization/autotune/export_utils.py`
   - 实际创建并插入 QuantizeLinear/DequantizeLinear 节点。

9. `modelopt/onnx/quantization/autotune/benchmark.py`
   - TensorRT Python API 和 `trtexec` Benchmark 实现。

10. `modelopt/onnx/quantization/autotune/common.py`
    - Region、InsertionScheme、PatternSchemes、PatternCache、Config 等基础数据结构。
