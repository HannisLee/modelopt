# Integrated AutoTune Q/DQ Chain Analysis

## 1. Executive Summary

**结论等级：A（核心 representation loss 已由源码明确证明）。** Integrated workflow 不会把 AutoTune 的 `optimized_final.onnx` 交给 ORT；它从无 Q/DQ 的输入图重新运行 ORT static quantization。AutoTune 的 `ResolvedInsertionPoint`（可表示具体 consumer input）先被压缩为 `covered: set[(node_index,input_index)]`，随后进一步投影为 node 名单、未量化输入三元组和 producer **op-type** 集合。OR T 根据这些约束重建 Q/DQ，而不是复制搜索模型中的 Q/DQ 节点。

因此最终 activation topology 是搜索 topology 的子集并不矛盾：精确 edge placement 已在 `get_ort_quantization_config()` 开始丢失，且 ORT 只为其量化 registry、node selection 和 output-quantization 策略可表达的部分重建 Q/DQ。`remove_partial_input_qdq()` 是后续明确删除/旁路 QDQ 的第二阶段，但静态代码不足以把给出的 36 个边全部唯一归因到它。

`weight_qdq 73 -> 97` 的直接机制是：covered input 会把 consumer **及其 producer** 加入 `nodes_to_quantize`；`quantize_static(..., per_channel=True)` 随后按 ORT 的 node-level QDQ 规则量化所选 MatMul/Gemm/Conv 的 initializer。`no_quantize_inputs` 仅为有普通 producer 的输入创建三元组，不能表示 initializer/Constant 权重，因此不能禁止该类 weight Q/DQ。增加 24 的精确逐节点归属仍属 topology 证据支持的推断。

## 2. Actual Source Call Chain

```text
modelopt/onnx/quantization/quantize.py:348 quantize()
  ↓ :116 _preprocess_onnx()（输入预处理；实际调用在 quantize 主体前段）
  ↓ :688 _find_nodes_to_quantize_autotune()
  ↓ :333 autotune.workflows.region_pattern_autotuning_workflow()
  ↓ autotuner_base.get_resolved_insertion_points()
  ↓ autotuner_base.py:481 get_ort_quantization_config()
  ↓ quantize.py:712 int8.quantize()
  ↓ int8.py:199 configure_ort()
  ↓ int8.py:278 ort_patching._quantize_static()/ORT quantize_static()
  ↓ int8.py:297 remove_partial_input_qdq()
  ↓ FP16 conversion / ONNX export
```

搜索阶段的 `export_onnx(best=True)` 可导出 exact Q/DQ 图；Integrated 路径却在 `quantize.py:345` 仅返回 config tuple，随后 `int8.quantize()` 重新从 `onnx_path` 加载无 Q/DQ 图（`int8.py:148-158`）。这已确认两阶段不是同一张 Q/DQ 图的连续变换。

## 3. Data Representation Transition

| Stage | Representation | Granularity |
|---|---|---|
| AutoTune | `ResolvedInsertionPoint` | tensor 或 `(consumer node, input index)` edge |
| conversion | `covered: set[(node_index,input_index)]` | consumer-input slot；producer/tensor identity 仅可反查 |
| ORT config | `nodes_to_quantize: list[str]` | node |
| ORT config | `no_quantize_inputs: (src,dst,tensor)` | 特定 edge，但只限 `inp.inputs` 有 producer |
| ORT config | `op_types_needing_output_quant: set[str]` | op-type，非 node/edge |
| ORT output | QDQ graph | ORT heuristic reconstruction |
| post-process | GraphSurgeon rewiring | 特定 matched QDQ edge |

## 4. `get_ort_quantization_config()` Deep Dive

源码：`autotuner_base.py:481-557`。

`covered` 的准确语义是“要求 Q/DQ 的 consumer input slot”。node-bound insertion point 直接加入 `(ip.node_index,ip.input_index)`；tensor-level point 遍历 `graph.tensor_users_map`，扩展为所有 consumer/input slot（:501-511）。它能表示 consumer 和 input index；在此集合内不保存 producer name/tensor object，尽管稍后能通过原图反查。**确认：它不是完整 Q/DQ node topology，也不含 scale、shared-pair identity、Q/DQ node naming 或每消费者是否 dedicated 的信息。**

`nodes_to_quantize` 首先取所有 covered consumer（:514），再把 covered input 的 producer 加入（:516-526）。注释明确原因：让 ORT 在 producer output 放 Q，例如 `Add -> Q/DQ -> Relu`。这是从 edge requirement 到 node inclusion 的扩张：若仅一个 input 被 covered，ORT 仍收到“量化整个 node”的选择。

`no_quantize_inputs`（:532-539）枚举这些 node 的未 covered input，但仅当 `inp.inputs` 非空才记录 `(producer,node,tensor)`。Initializer/Constant 没有普通 producer，不能进入该列表。`op_types_needing_output_quant`（:542-549）只在 covered consumer 是 `get_activation_ops()` 成员时收集其 producer 的 **op type**；多个同 type producer 因而不可区分。

## 5. Activation Q/DQ Loss Analysis

已知 122→86 和 `FINAL ONLY=0` 与此设计一致，但“每条边在哪一行被删”并不能只由静态源码唯一证明。

`configure_ort()`（`ort_utils.py:598-678`）注册/删除 ORT QDQ op registry，并构造 `OpTypesToExcludeOutputQuantization`：除 `op_types_needing_output_quant` 外的 op type 默认不请求 output QDQ（:648-656）。它设置 `ActivationSymmetric=True`、`DedicatedQDQPair=False`、`ForceQuantizeNoInputCheck=True`、`AddQDQPairToWeight=True`（:662-681）。因此 shared pair 是 ORT 的全局策略，不是 AutoTune 的 exact shared-pair identity。

`remove_partial_input_qdq()`（`graph_utils.py:658-730`）只处理 `no_quantize_inputs` 标记的边。它从 source 追 `source.o().o()`，确认 DQ，随后按 **目标 node 名和 DQ 输出 tensor** 定位指定 consumer input 并直接接回 `source.outputs[0]`（:670-710）。注释明确处理 `DedicatedQDQPair=False` 的共享 pair，避免误删其它 branch。故：它确实能删除原先由 ORT 插入的、未覆盖输入上的 QDQ；但只有相关 edge 进入 `no_quantize_inputs` 才会发生。静态源码不能确认所列 `Cast->Sqrt`、`Div->Sqrt*`、`Add->GELU`、final Add->LayerNorm 是否均进入该列表。

`Sqrt` 和 GELU 不在 `int8.quantize()` docstring 的 supported op 列表（`int8.py:137-140`）；AutoTune 可以在任何 resolved edge 插 Q/DQ，而 ORT 的 node/registry reconstruction 不保证为非 quantizable consumer 保持该 input boundary。对 12×Cast→Sqrt、12×shared Div→Sqrt_1/_2、11×Add→GELU 和 final Add→LayerNorm，**CONFIRMED primary explanation** 是边级需求没有被作为 exact boundary 传给 ORT；是否额外由 `no_quantize_inputs` 后处理删除，为 STRONGLY SUPPORTED / per-edge UNKNOWN。

`get_concat_eliminated_tensors()` 只在 `passes` 含 `concat_elimination` 时调用（`int8.py:257-260`），仅构造 `group_qdq_tensors` 供 ORT grouping；它不按上述 Sqrt/GELU/LayerNorm 边名做删除，不能单独解释该规律（CONFIRMED exclusion for these named non-Concat edges）。

## 6. Weight Q/DQ Increase Analysis

**CONFIRMED source chain：** covered slot → producer/consumer 都加入 `nodes_to_quantize`（`autotuner_base.py:514-526`）→ `int8.quantize()` 将该 list 直接传给 `quantize_static`（`int8.py:255-289`）→ `per_channel=True`（:282）和 `AddQDQPairToWeight=True`（`ort_utils.py:668`）使 ORT 为所选 quantizable node 的 weights 构造 Q/DQ。

对问题 1：源码证明 selected node 被交给 ORT node-level static QDQ；ORT 对可量化 MatMul/Gemm/Conv 的 initializer weight 生成量化表示是此 API 的设计结果。对问题 2：`no_quantize_inputs` 无法表示 weight initializer（见上），且 post-process 只绕过 source-node→target 的 activation edge；不能阻止 weight QDQ。对问题 3：这强烈支持 73→97 是 expanded `nodes_to_quantize` 使更多 weight-bearing node 被 ORT 量化；**24 的精确来源需要把两图 weight initializer consumer 映射到 node list，当前静态代码本身不能计数证明。**

## 7. Exact Point Where Topology Diverges

**Primary cause (CONFIRMED)：** `autotuner_base.py:get_ort_quantization_config()`：exact resolved insertion points 没有导出为 Q/DQ graph，而是转换为 ORT config tuple。最先不可逆的投影是 `covered` 再到 `nodes_to_quantize` / producer op-type set。

**Secondary transformation (CONFIRMED)：** ORT `quantize_static` 从原始图重建 Q/DQ，受 registry、node list、output exclusion、`DedicatedQDQPair=False` 和 per-channel weight 策略控制。

**Post-processing effect (CONFIRMED mechanism; per-edge UNKNOWN)：** `remove_partial_input_qdq()` 可旁路 `no_quantize_inputs` 中的 ORT QDQ。

## 8. Confirmed vs Inferred Findings

| Finding | Status | Evidence |
|---|---|---|
| Integrated 不使用 `optimized_final.onnx` 作为 ORT 输入 | CONFIRMED | `quantize.py:345,712`; `int8.py:148` |
| exact edge→node-level config 投影 | CONFIRMED | `autotuner_base.py:497-557` |
| producer inclusion 扩张 selected node set | CONFIRMED | `autotuner_base.py:516-526` |
| weight input 不能进 `no_quantize_inputs` | CONFIRMED | `autotuner_base.py:532-539` |
| ORT 可新增 weight QDQ | CONFIRMED | `int8.py:278-289`, `ort_utils.py:668` |
| 36 条具体 activation 边均由 post-process 删除 | UNKNOWN | 需运行时记录 `no_quantize_inputs` / ORT intermediate graph |
| 24 条新增 weight 的逐节点归因 | STRONGLY SUPPORTED | node expansion + observed topology；缺 node mapping |

## 9. Final Root-Cause Statement

AutoTune benchmark 的一致性标准是 exact edge topology：`producer tensor -> Q/DQ -> consumer.input[index]`。Integrated API 当前仅保留到“哪些 node 被量化、哪些 producer op type 要 output quant、哪些有 producer 的 input 要绕过”的级别，并让 ORT 重建图；它不保留 AutoTune 的 dedicated/shared QDQ identity 或任意非量化 op input edge。故即使 calibration scale 完全正确，最终 calibrated model 仍可有不同 Q/DQ placement，AutoTune benchmark latency 不能自动代表最终 calibrated ONNX latency。

数量一致 < quantized node set 一致 < exact edge topology 一致；本 workflow 在源码上保证的至多是受 ORT constraints 约束的 node-level intent，而非第三种一致性。
