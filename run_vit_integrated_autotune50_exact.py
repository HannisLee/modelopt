#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the ViT INT8 integrated AutoTune-50 experiment with the local source tree."""

import inspect
import sys
from pathlib import Path

import numpy as np
import onnx


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from modelopt.onnx.quantization import quantize  # noqa: E402
from modelopt.onnx.quantization.autotune.workflows import (  # noqa: E402
    benchmark_onnx_model,
    init_benchmark_instance,
)
from modelopt.onnx.quantization.qdq_calibration import get_qdq_topology  # noqa: E402


MODEL_PATH = REPO_ROOT / "model_home/vit/vit.fp16.onnx"
CALIBRATION_PATH = REPO_ROOT / "model_home/vit/calib.fp16.npy"
OUTPUT_PATH = REPO_ROOT / "model_home/vit/vit.int8.integrated-exact-autotune50.onnx"
AUTOTUNE_OUTPUT_DIR = REPO_ROOT / "autotune_output/vit/integrated-exact-s50"
TIMING_CACHE_PATH = AUTOTUNE_OUTPUT_DIR / "trt_timing.cache"
BENCHMARK_LOG_PATH = AUTOTUNE_OUTPUT_DIR / "final_exact_calibrated.log"


def main() -> None:
    quantize_source = Path(inspect.getsourcefile(quantize)).resolve()
    expected_source = REPO_ROOT / "modelopt/onnx/quantization/quantize.py"
    if quantize_source != expected_source:
        raise RuntimeError(
            f"Expected local source '{expected_source}', but imported '{quantize_source}'."
        )
    if AUTOTUNE_OUTPUT_DIR.exists() or OUTPUT_PATH.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing experiment. Remove or rename these paths first: "
            f"{AUTOTUNE_OUTPUT_DIR}, {OUTPUT_PATH}"
        )

    AUTOTUNE_OUTPUT_DIR.mkdir(parents=True)
    calibration_data = np.load(CALIBRATION_PATH, mmap_mode="r")
    print(f"Using local quantize source: {quantize_source}")
    print(f"Calibration data: {calibration_data.shape}, {calibration_data.dtype}")

    quantize(
        str(MODEL_PATH),
        quantize_mode="int8",
        calibration_data=calibration_data,
        output_path=str(OUTPUT_PATH),
        autotune=True,
        autotune_output_dir=str(AUTOTUNE_OUTPUT_DIR),
        autotune_num_schemes_per_region=50,
        autotune_warmup_runs=10,
        autotune_timing_runs=50,
        autotune_timing_cache=str(TIMING_CACHE_PATH),
        calibration_eps=["cpu"],
        high_precision_dtype="fp16",
        log_level="INFO",
    )

    output_model = onnx.load(OUTPUT_PATH, load_external_data=True)
    topology = get_qdq_topology(output_model)
    q_count = len(topology.q_nodes)
    dq_count = len(topology.dq_nodes)
    print(f"Final QDQ topology: Q={q_count}, DQ={dq_count}, exact edges={len(topology.edges)}")

    init_benchmark_instance(
        timing_cache_file=str(TIMING_CACHE_PATH),
        warmup_runs=10,
        timing_runs=50,
    )
    latency_ms = benchmark_onnx_model(str(OUTPUT_PATH), str(BENCHMARK_LOG_PATH))
    if latency_ms == float("inf"):
        raise RuntimeError("TensorRT failed to build or benchmark the final exact-edge model")
    print(f"Final TensorRT median latency: {latency_ms:.6f} ms")


if __name__ == "__main__":
    main()
