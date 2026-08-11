# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Calibrate an existing QDQ graph without changing its QDQ topology."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationMethod
from onnxruntime.quantization.calibrate import CalibrationDataReader, TensorsData
from onnxruntime.quantization.quant_utils import (
    compute_data_quant_params,
    compute_scale_zp,
    get_qmin_qmax_for_qType,
)

from modelopt.onnx.quantization.ort_patching import collect_tensor_ranges


@dataclass(frozen=True)
class QDQTopology:
    """Stable description of the QDQ nodes and exact consumer input slots in a graph."""

    q_nodes: tuple[str, ...]
    dq_nodes: tuple[str, ...]
    edges: tuple[tuple[str, str, str, str, int], ...]


def get_qdq_topology(model: onnx.ModelProto) -> QDQTopology:
    """Return the exact QDQ topology for comparison before and after calibration."""
    tensor_consumers: dict[str, list[tuple[onnx.NodeProto, int]]] = defaultdict(list)
    for node in model.graph.node:
        for input_index, tensor_name in enumerate(node.input):
            tensor_consumers[tensor_name].append((node, input_index))

    q_nodes = [node for node in model.graph.node if node.op_type == "QuantizeLinear"]
    dq_nodes = [node for node in model.graph.node if node.op_type == "DequantizeLinear"]
    _validate_qdq_node_names(q_nodes, "QuantizeLinear")
    _validate_qdq_node_names(dq_nodes, "DequantizeLinear")

    edges = []
    for q_node in q_nodes:
        dq_consumers = [
            node
            for node, _ in tensor_consumers.get(q_node.output[0], [])
            if node.op_type == "DequantizeLinear"
        ]
        if len(dq_consumers) != 1:
            raise ValueError(
                f"Expected one DequantizeLinear consumer for '{q_node.name}', "
                f"got {len(dq_consumers)}"
            )

        dq_node = dq_consumers[0]
        data_consumers = tensor_consumers.get(dq_node.output[0], [])
        if not data_consumers:
            edges.append((q_node.input[0], q_node.name, dq_node.name, "", -1))
        else:
            edges.extend(
                (q_node.input[0], q_node.name, dq_node.name, consumer.name, input_index)
                for consumer, input_index in data_consumers
            )

    return QDQTopology(
        q_nodes=tuple(sorted(node.name for node in q_nodes)),
        dq_nodes=tuple(sorted(node.name for node in dq_nodes)),
        edges=tuple(sorted(edges)),
    )


def calibrate_qdq_model(
    qdq_model: onnx.ModelProto,
    calibration_model: str | Path | onnx.ModelProto,
    calibration_data_reader: CalibrationDataReader | None,
    calibration_method: CalibrationMethod,
    *,
    calibration_extra_options: dict | None = None,
    calibration_cache_scales: dict[str, float] | None = None,
    use_external_data_format: bool = False,
) -> onnx.ModelProto:
    """Update QDQ parameters in-place while preserving every QDQ node and data edge."""
    model = onnx.ModelProto()
    model.CopyFrom(qdq_model)
    topology_before = get_qdq_topology(model)

    initializer_map = {initializer.name: initializer for initializer in model.graph.initializer}
    tensor_consumers = _build_tensor_consumers(model)
    q_nodes = [node for node in model.graph.node if node.op_type == "QuantizeLinear"]

    activation_tensor_names = {
        node.input[0] for node in q_nodes if node.input[0] not in initializer_map
    }
    tensor_ranges = _collect_activation_ranges(
        calibration_model,
        calibration_data_reader,
        activation_tensor_names,
        calibration_method,
        calibration_extra_options,
        calibration_cache_scales,
        use_external_data_format,
    )

    for q_node in q_nodes:
        dq_node = _get_dq_consumer(q_node, tensor_consumers)
        source_name = q_node.input[0]
        q_scale_initializer, q_zero_point_initializer = _get_qdq_initializers(
            q_node, initializer_map
        )

        axis = None
        if source_name in initializer_map:
            axis = _get_weight_axis(dq_node, tensor_consumers)
            zero_point, scale = _compute_initializer_params(
                initializer_map[source_name], q_zero_point_initializer.data_type, axis
            )
        else:
            zero_point, scale = _compute_activation_params(
                source_name,
                tensor_ranges,
                calibration_cache_scales,
                q_zero_point_initializer.data_type,
            )

        _replace_initializer(
            q_scale_initializer.name,
            _cast_scale(scale, q_scale_initializer, axis),
            initializer_map,
        )
        _replace_initializer(
            q_zero_point_initializer.name,
            _cast_zero_point(zero_point, q_zero_point_initializer, axis),
            initializer_map,
        )
        _set_axis(q_node, axis)

        dq_scale_initializer, dq_zero_point_initializer = _get_qdq_initializers(
            dq_node, initializer_map
        )
        _replace_initializer(
            dq_scale_initializer.name,
            _cast_scale(scale, dq_scale_initializer, axis),
            initializer_map,
        )
        _replace_initializer(
            dq_zero_point_initializer.name,
            _cast_zero_point(zero_point, dq_zero_point_initializer, axis),
            initializer_map,
        )
        _set_axis(dq_node, axis)

    topology_after = get_qdq_topology(model)
    if topology_after != topology_before:
        raise RuntimeError("QDQ topology changed while updating calibration parameters")

    onnx.checker.check_model(model)
    return model


def _validate_qdq_node_names(nodes: list[onnx.NodeProto], op_type: str) -> None:
    names = [node.name for node in nodes]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError(f"All {op_type} nodes must have unique, non-empty names")


def _build_tensor_consumers(
    model: onnx.ModelProto,
) -> dict[str, list[tuple[onnx.NodeProto, int]]]:
    consumers: dict[str, list[tuple[onnx.NodeProto, int]]] = defaultdict(list)
    for node in model.graph.node:
        for input_index, tensor_name in enumerate(node.input):
            consumers[tensor_name].append((node, input_index))
    return consumers


def _get_dq_consumer(
    q_node: onnx.NodeProto,
    tensor_consumers: dict[str, list[tuple[onnx.NodeProto, int]]],
) -> onnx.NodeProto:
    consumers = tensor_consumers.get(q_node.output[0], [])
    dq_nodes = [node for node, _ in consumers if node.op_type == "DequantizeLinear"]
    if len(consumers) != 1 or len(dq_nodes) != 1:
        raise ValueError(f"'{q_node.name}' must feed exactly one DequantizeLinear node")
    return dq_nodes[0]


def _get_qdq_initializers(
    node: onnx.NodeProto,
    initializer_map: dict[str, onnx.TensorProto],
) -> tuple[onnx.TensorProto, onnx.TensorProto]:
    if len(node.input) < 3:
        raise ValueError(f"QDQ node '{node.name}' must have explicit scale and zero-point inputs")
    try:
        return initializer_map[node.input[1]], initializer_map[node.input[2]]
    except KeyError as error:
        raise ValueError(f"QDQ parameters for '{node.name}' must be graph initializers") from error


def _collect_activation_ranges(
    calibration_model: str | Path | onnx.ModelProto,
    calibration_data_reader: CalibrationDataReader | None,
    activation_tensor_names: set[str],
    calibration_method: CalibrationMethod,
    calibration_extra_options: dict | None,
    calibration_cache_scales: dict[str, float] | None,
    use_external_data_format: bool,
) -> TensorsData | None:
    if not activation_tensor_names or calibration_cache_scales is not None:
        return None
    if calibration_data_reader is None:
        raise ValueError("Runtime calibration requires a calibration data reader")
    return collect_tensor_ranges(
        calibration_model,
        calibration_data_reader,
        activation_tensor_names,
        calibrate_method=calibration_method,
        use_external_data_format=use_external_data_format,
        extra_options=calibration_extra_options,
    )


def _compute_activation_params(
    tensor_name: str,
    tensor_ranges: TensorsData | None,
    calibration_cache_scales: dict[str, float] | None,
    quant_type: int,
) -> tuple[np.ndarray, np.ndarray]:
    if calibration_cache_scales is not None:
        scale = calibration_cache_scales.get(tensor_name)
        if scale is None:
            scale = calibration_cache_scales.get(f"{tensor_name}_scale")
        if scale is None:
            raise ValueError(f"Calibration cache has no scale for '{tensor_name}'")
        zero_point_dtype = onnx.helper.tensor_dtype_to_np_dtype(quant_type)
        return np.asarray(0, dtype=zero_point_dtype), np.asarray(scale, dtype=np.float32)

    if tensor_ranges is None or tensor_name not in tensor_ranges:
        raise ValueError(f"Calibration produced no range for '{tensor_name}'")
    tensor_range = tensor_ranges[tensor_name]
    qmin, qmax = get_qmin_qmax_for_qType(quant_type, reduce_range=False, symmetric=True)
    zero_point, scale = compute_scale_zp(
        tensor_range.range_value[0],
        tensor_range.range_value[1],
        qmin,
        qmax,
        symmetric=True,
    )
    return zero_point, scale


def _compute_initializer_params(
    initializer: onnx.TensorProto,
    quant_type: int,
    axis: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(onnx.numpy_helper.to_array(initializer), dtype=np.float32)
    if axis is None:
        return compute_data_quant_params(data.ravel(), quant_type, symmetric=True)

    zero_points = []
    scales = []
    for channel_index in range(data.shape[axis]):
        zero_point, scale = compute_data_quant_params(
            data.take(channel_index, axis).ravel(), quant_type, symmetric=True
        )
        zero_points.append(zero_point)
        scales.append(scale)
    return np.asarray(zero_points).reshape(-1), np.asarray(scales).reshape(-1)


def _get_weight_axis(
    dq_node: onnx.NodeProto,
    tensor_consumers: dict[str, list[tuple[onnx.NodeProto, int]]],
) -> int | None:
    axes = set()
    for consumer, input_index in tensor_consumers.get(dq_node.output[0], []):
        axis = _get_consumer_weight_axis(consumer, input_index)
        if axis is None:
            return None
        axes.add(axis)
    return axes.pop() if len(axes) == 1 else None


def _get_consumer_weight_axis(node: onnx.NodeProto, input_index: int) -> int | None:
    if input_index != 1:
        return None
    if node.op_type == "Conv":
        return 0
    if node.op_type == "ConvTranspose":
        return 1
    if node.op_type == "MatMul":
        return 1
    if node.op_type == "Gemm":
        trans_b = next(
            (attribute.i for attribute in node.attribute if attribute.name == "transB"), 0
        )
        return 0 if trans_b else 1
    return None


def _cast_scale(
    value: np.ndarray,
    initializer: onnx.TensorProto,
    axis: int | None,
) -> np.ndarray:
    dtype = onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type)
    value_array = np.asarray(value)
    if not np.all(np.isfinite(value_array)) or np.any(value_array <= 0):
        raise ValueError(f"Invalid scale values for '{initializer.name}'")
    if np.issubdtype(dtype, np.floating):
        value_array = np.maximum(value_array, np.finfo(dtype).smallest_subnormal)
    array = np.asarray(value_array, dtype=dtype)
    if axis is None:
        array = array.reshape(onnx.numpy_helper.to_array(initializer).shape)
    return array


def _cast_zero_point(
    value: np.ndarray,
    initializer: onnx.TensorProto,
    axis: int | None,
) -> np.ndarray:
    dtype = onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type)
    array = np.asarray(value, dtype=dtype)
    if axis is None:
        array = array.reshape(onnx.numpy_helper.to_array(initializer).shape)
    return array


def _replace_initializer(
    name: str,
    value: np.ndarray,
    initializer_map: dict[str, onnx.TensorProto],
) -> None:
    new_initializer = onnx.numpy_helper.from_array(value, name)
    initializer_map[name].CopyFrom(new_initializer)


def _set_axis(node: onnx.NodeProto, axis: int | None) -> None:
    attributes = [attribute for attribute in node.attribute if attribute.name != "axis"]
    del node.attribute[:]
    node.attribute.extend(attributes)
    if axis is not None:
        node.attribute.append(onnx.helper.make_attribute("axis", axis))
