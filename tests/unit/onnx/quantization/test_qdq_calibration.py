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

import numpy as np
import onnx
import pytest
from onnxruntime.quantization import CalibrationMethod

from modelopt.onnx.quantization.qdq_calibration import calibrate_qdq_model, get_qdq_topology


def _make_qdq_pair(
    source_name: str,
    suffix: str,
) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto], str]:
    q_output = f"{suffix}_quantized"
    dq_output = f"{suffix}_dequantized"
    q_scale = f"q_scale_{suffix}"
    q_zero_point = f"q_zero_point_{suffix}"
    dq_scale = f"dq_scale_{suffix}"
    dq_zero_point = f"dq_zero_point_{suffix}"
    nodes = [
        onnx.helper.make_node(
            "QuantizeLinear",
            [source_name, q_scale, q_zero_point],
            [q_output],
            name=f"Q_{suffix}",
        ),
        onnx.helper.make_node(
            "DequantizeLinear",
            [q_output, dq_scale, dq_zero_point],
            [dq_output],
            name=f"DQ_{suffix}",
        ),
    ]
    initializers = [
        onnx.numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), q_scale),
        onnx.numpy_helper.from_array(np.asarray([0], dtype=np.int8), q_zero_point),
        onnx.numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), dq_scale),
        onnx.numpy_helper.from_array(np.asarray([0], dtype=np.int8), dq_zero_point),
    ]
    return nodes, initializers, dq_output


def _make_matmul_qdq_model() -> onnx.ModelProto:
    activation_nodes, activation_parameters, activation_dq = _make_qdq_pair("x", "x")
    weight_nodes, weight_parameters, weight_dq = _make_qdq_pair("weight", "weight")
    nodes = [
        *activation_nodes,
        *weight_nodes,
        onnx.helper.make_node("MatMul", [activation_dq, weight_dq], ["y"], name="MatMul_0"),
        onnx.helper.make_node("Identity", ["x"], ["x_identity"], name="Identity_0"),
    ]
    initializers = [
        onnx.numpy_helper.from_array(
            np.asarray([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], dtype=np.float32),
            "weight",
        ),
        *activation_parameters,
        *weight_parameters,
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "matmul_qdq",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2])],
        [
            onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3]),
            onnx.helper.make_tensor_value_info("x_identity", onnx.TensorProto.FLOAT, [1, 2]),
        ],
        initializers,
    )
    return onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 19)])


def _initializer_array(model: onnx.ModelProto, name: str) -> np.ndarray:
    initializer = next(
        initializer for initializer in model.graph.initializer if initializer.name == name
    )
    return onnx.numpy_helper.to_array(initializer)


def _axis(node: onnx.NodeProto) -> int | None:
    return next((attribute.i for attribute in node.attribute if attribute.name == "axis"), None)


def test_calibrate_qdq_model_preserves_exact_edges_and_uses_per_channel_weights():
    model = _make_matmul_qdq_model()
    topology_before = get_qdq_topology(model)

    calibrated = calibrate_qdq_model(
        model,
        model,
        calibration_data_reader=None,
        calibration_method=CalibrationMethod.Entropy,
        calibration_cache_scales={"x_scale": 0.25},
    )

    assert get_qdq_topology(calibrated) == topology_before
    assert _initializer_array(calibrated, "q_scale_x").item() == pytest.approx(0.25)
    assert _initializer_array(calibrated, "dq_scale_x").item() == pytest.approx(0.25)

    q_weight_scale = _initializer_array(calibrated, "q_scale_weight")
    dq_weight_scale = _initializer_array(calibrated, "dq_scale_weight")
    assert q_weight_scale.shape == (3,)
    np.testing.assert_array_equal(q_weight_scale, dq_weight_scale)
    assert np.all(q_weight_scale > 0)

    nodes = {node.name: node for node in calibrated.graph.node}
    assert _axis(nodes["Q_weight"]) == 1
    assert _axis(nodes["DQ_weight"]) == 1
    assert _axis(nodes["Q_x"]) is None
    assert _axis(nodes["DQ_x"]) is None

    identity_node = nodes["Identity_0"]
    assert identity_node.input[0] == "x"


def test_calibrate_qdq_model_rejects_missing_activation_cache_scale():
    model = _make_matmul_qdq_model()
    with pytest.raises(ValueError, match="Calibration cache has no scale for 'x'"):
        calibrate_qdq_model(
            model,
            model,
            calibration_data_reader=None,
            calibration_method=CalibrationMethod.Entropy,
            calibration_cache_scales={},
        )
