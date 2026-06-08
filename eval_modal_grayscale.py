"""
Modal A100 evaluator for the grayscale kernel task.

Deploy once:
    uv run modal deploy eval_modal_grayscale.py

Then run_eval.py calls evaluate_kernel.remote(kernel_code).
"""

import modal

BENCHMARK_SIZES = [512, 1024, 2048, 4096, 8192, 16384]
TEST_SIZES = [512, 1024, 2048]

app = modal.App("grayscale-kernel-eval")

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .pip_install("triton")
)


@app.function(gpu="A100", image=image, timeout=300)
def evaluate_kernel(kernel_code: str, warmup_iters: int = 5, eval_iters: int = 20) -> str:
    import json
    import math
    import platform
    import traceback
    import types

    import torch

    def ref_kernel(data):
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=data.device, dtype=data.dtype)
        return torch.sum(data * weights, dim=-1)

    def generate_input(size: int, seed: int):
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
        return torch.rand(size, size, 3, device="cuda", dtype=torch.float32, generator=gen).contiguous()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    torch_ver = torch.__version__
    plat = platform.platform()

    result = {
        "success": False,
        "tests_passed": 0,
        "tests_total": len(TEST_SIZES),
        "test_details": [],
        "benchmark": None,
        "benchmark_details": [],
        "gpu_name": gpu_name,
        "torch_version": torch_ver,
        "platform": plat,
        "error": None,
    }

    module = types.ModuleType("submission")
    module.__dict__["__builtins__"] = __builtins__
    try:
        exec(kernel_code, module.__dict__)
    except Exception:
        result["error"] = f"Failed to load submission:\n{traceback.format_exc()}"
        return json.dumps(result)

    custom_kernel = getattr(module, "custom_kernel", None)
    if custom_kernel is None:
        result["error"] = "submission.py does not define custom_kernel(data)"
        return json.dumps(result)

    # Correctness tests
    for size in TEST_SIZES:
        data = generate_input(size, seed=42)
        detail = {"size": size, "passed": False, "error": None}
        try:
            expected = ref_kernel(data.clone())
            actual = custom_kernel(data)
            torch.cuda.synchronize()
            if actual.shape != expected.shape:
                detail["error"] = f"shape mismatch: got {tuple(actual.shape)}, expected {tuple(expected.shape)}"
            elif not torch.allclose(actual.float(), expected.float(), rtol=1e-4, atol=1e-4):
                max_diff = (actual.float() - expected.float()).abs().max().item()
                detail["error"] = f"values differ, max abs diff: {max_diff:.6f}"
            else:
                detail["passed"] = True
                result["tests_passed"] += 1
        except Exception:
            detail["error"] = traceback.format_exc()[:600]
        result["test_details"].append(detail)

    if result["tests_passed"] < result["tests_total"]:
        result["error"] = "Correctness check failed — see test_details"
        return json.dumps(result)

    if warmup_iters == 0 and eval_iters == 0:
        result["success"] = True
        return json.dumps(result)

    # Benchmarks
    times_us = []
    try:
        # GPU warmup: run first shape for ≥2 s to reach boost clock
        _wm = BENCHMARK_SIZES[0]
        import time as _time
        _t0 = _time.time()
        _wi = 0
        while _time.time() - _t0 < 2.0:
            custom_kernel(generate_input(_wm, seed=99999 + _wi))
            torch.cuda.synchronize()
            _wi += 1

        for size in BENCHMARK_SIZES:
            data = generate_input(size, seed=0)

            for _ in range(warmup_iters):
                custom_kernel(data)
            torch.cuda.synchronize()

            iter_times = []
            for i in range(eval_iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                custom_kernel(data)
                end.record()
                torch.cuda.synchronize()
                iter_times.append(start.elapsed_time(end) * 1000.0)  # ms → µs

            mean_us = sum(iter_times) / len(iter_times)
            variance = sum((t - mean_us) ** 2 for t in iter_times) / len(iter_times)
            stderr_us = variance ** 0.5 / math.sqrt(len(iter_times))

            result["benchmark_details"].append({
                "size": size,
                "mean_us": round(mean_us, 3),
                "min_us": round(min(iter_times), 3),
                "max_us": round(max(iter_times), 3),
                "stderr_us": round(stderr_us, 3),
            })
            times_us.append(mean_us)

        geomean = math.exp(sum(math.log(t) for t in times_us) / len(times_us))
        result["benchmark"] = {"geomean_us": round(geomean, 3)}
        result["success"] = True

    except Exception:
        result["error"] = f"Benchmark failed:\n{traceback.format_exc()}"

    return json.dumps(result)
