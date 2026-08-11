from pathlib import Path

import onnx
import yaml

from modelopt.onnx.quantization.autotune import Config, QDQAutotuner
from modelopt.onnx.quantization.autotune.workflows import (
    benchmark_onnx_model,
    init_benchmark_instance,
)


CONFIG_PATH = Path(__file__).with_name("autotune.yaml")


def resolve_path(config_dir: Path, path: str) -> Path:
    """将 YAML 中的相对路径转换为相对配置文件的绝对路径。"""
    return (config_dir / path).resolve()


with CONFIG_PATH.open(encoding="utf-8") as file:
    settings = yaml.safe_load(file)

config_dir = CONFIG_PATH.parent
paths = settings["paths"]
quantization = settings["quantization"]
autotune_settings = settings["autotune"]
model_path = resolve_path(config_dir, paths["model"])
output_dir = resolve_path(config_dir, paths["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)
schemes_per_region = autotune_settings["schemes_per_region"][0]

print(f"已读取配置文件：{CONFIG_PATH}")
print(f"待优化模型：{model_path}")
print(f"输出目录：{output_dir}")

# 初始化全局 benchmark（benchmark_onnx_model 运行前必须调用）。
init_benchmark_instance(
    use_trtexec=autotune_settings["use_trtexec"],
    timing_cache_file=str(output_dir / "timing.cache"),
    warmup_runs=autotune_settings["warmup_runs"],
    timing_runs=autotune_settings["timing_runs"],
)

# 加载模型。
model = onnx.load(model_path)

# 初始化 autotuner，并自动发现优化 region。
autotuner = QDQAutotuner(model)
config = Config(default_quant_type=quantization["mode"], verbose=True)
autotuner.initialize(config)

# 测量 baseline（不插入 Q/DQ）。
baseline_path = output_dir / "baseline.onnx"
autotuner.export_onnx(str(baseline_path), insert_qdq=False)
baseline_latency = benchmark_onnx_model(str(baseline_path))
autotuner.submit(baseline_latency)
print(f"基线延迟：{baseline_latency:.2f} ms")

# 逐个 profile region。
regions = autotuner.regions
print(f"发现 {len(regions)} 个待优化 region")

for region_idx, region in enumerate(regions):
    print(f"\n处理 region {region_idx + 1}/{len(regions)}")

    # 设置当前 profile region。
    autotuner.set_profile_region(region, commit=(region_idx > 0))

    # 若该 pattern 已被 profile，则没有新 scheme 可生成。
    if autotuner.current_profile_pattern_schemes is None:
        print("  该 pattern 已完成，跳过。")
        continue

    # 生成并测试 YAML 中指定数量的 scheme。
    for scheme_num in range(schemes_per_region):
        scheme_idx = autotuner.generate()

        if scheme_idx == -1:
            print(f"  在第 {scheme_num} 个候选后没有更多唯一 scheme。")
            break

        # 导出带 Q/DQ 节点的模型并测量性能。
        model_bytes = autotuner.export_onnx(None, insert_qdq=True)
        latency = benchmark_onnx_model(model_bytes)
        success = latency != float("inf")
        autotuner.submit(latency, success=success)

        if success:
            speedup = baseline_latency / latency
            print(f"  方案 {scheme_idx}: {latency:.2f} ms（{speedup:.3f}x）")
        else:
            print(f"  方案 {scheme_idx}: 测量失败。")

    # 最优 scheme 会由 autotuner 自动选择。
    pattern_schemes = autotuner.current_profile_pattern_schemes
    if pattern_schemes and pattern_schemes.best_scheme:
        print(f"  当前最优：{pattern_schemes.best_scheme.latency_ms:.2f} ms")

# 提交最后一个 region。
autotuner.set_profile_region(None, commit=True)

# 导出优化后的模型。
final_path = output_dir / "optimized_final.onnx"
autotuner.export_onnx(str(final_path), insert_qdq=True)
print(f"\n优化完成，最终模型：{final_path}")
