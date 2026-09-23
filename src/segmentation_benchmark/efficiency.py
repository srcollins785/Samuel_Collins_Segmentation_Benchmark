"""Sections 30 to 33: parameters, size, training time, latency and memory.

Section 29 is explicit that a measurement must carry its conditions - model
variant, seed, input resolution, batch size, precision, hardware and run
identifier - or it cannot be compared with anything. Every function here
returns those alongside the number.

Section 33 asks for warm-up before timing, evaluation mode without gradient
tracking, device synchronization around timed regions, batch-one latency
distinguished from batched throughput, and a statement of what the timed
region includes. All of that is implemented rather than assumed, because on
an asynchronous backend a timer without synchronization measures how long it
took to *enqueue* the work and reliably reports a model as several times
faster than it is.

One honest limitation, stated here and repeated in the results. There is no
CUDA on this hardware, so Section 29's peak GPU memory is measured through
``torch.mps.current_allocated_memory``, which reports the allocator's
current total rather than a driver-level peak. It is a weaker measurement
than ``torch.cuda.max_memory_allocated`` and it is labeled as such
everywhere it appears. It is sampled immediately after a forward pass, which
is where the peak occurs.
"""

import json
import platform
import time

import torch

from ._config import (
    BYTES_PER_MB,
    INFERENCE_BATCH_SIZES,
    INFERENCE_TIMED_IMAGES,
    INFERENCE_WARMUP_ITERS,
    INPUT_SIZE,
    SEED,
)
from .train import memory_allocated_mb, reset_memory_stats, synchronize


def environment() -> dict:
    """The hardware and software a set of measurements was produced on."""
    import numpy
    import torchvision

    record = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "seed": SEED,
        "precision": "float32",
        "mixed_precision": False,
    }

    if torch.cuda.is_available():
        record["device"] = torch.cuda.get_device_name(0)
        record["backend"] = "cuda"
        record["memory_metric"] = "torch.cuda.max_memory_allocated (true peak)"
    elif torch.backends.mps.is_available():
        # platform.processor() returns 'arm' on Apple silicon, which does
        # not identify the chip. sysctl does.
        try:
            import subprocess
            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except Exception:
            chip = "Apple silicon"
        record["device"] = chip
        record["backend"] = "mps"
        record["memory_metric"] = (
            "torch.mps.current_allocated_memory (allocator total, not a "
            "driver-level peak; weaker than the CUDA equivalent)"
        )
    else:
        record["device"] = "cpu"
        record["backend"] = "cpu"
        record["memory_metric"] = "not measured on CPU"

    return record


def complexity(model, input_size: int = INPUT_SIZE) -> dict:
    """Multiply-accumulate count, via fvcore with thop as a fallback.

    Section 30 warns that a parameter count is not a substitute for measured
    time or memory, which is true in both directions: FLOPs are not either.
    A dilated ResNet50 and a plain one have nearly identical parameter
    counts and very different costs, because dilation keeps the feature map
    large. That gap is what this measures and why it is worth reporting
    next to the parameter count rather than instead of it.

    fvcore is tried first because it reports which operators it could not
    account for, so an underestimate is visible rather than silent.
    """
    example = torch.randn(1, 3, input_size, input_size)
    model = model.eval().cpu()

    try:
        from fvcore.nn import FlopCountAnalysis
        import logging
        logging.getLogger("fvcore").setLevel(logging.ERROR)

        analysis = FlopCountAnalysis(model, example)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        total = analysis.total()
        skipped = sorted(analysis.unsupported_ops().keys())
        return {
            "gmacs": total / 1e9,
            "complexity_tool": "fvcore",
            "unsupported_operators": skipped,
            "complexity_note": (
                "multiply-accumulates for one forward pass at "
                f"{input_size}x{input_size}, batch 1"
                + (f"; {len(skipped)} operator types unaccounted for"
                   if skipped else "")
            ),
        }
    except Exception as fvcore_error:
        try:
            from thop import profile
            macs, _ = profile(model, inputs=(example,), verbose=False)
            return {
                "gmacs": macs / 1e9,
                "complexity_tool": "thop",
                "unsupported_operators": [],
                "complexity_note": (
                    f"thop fallback; fvcore failed: {type(fvcore_error).__name__}"
                ),
            }
        except Exception as thop_error:
            return {
                "gmacs": None,
                "complexity_tool": None,
                "unsupported_operators": [],
                "complexity_note": (
                    f"not measured: fvcore {type(fvcore_error).__name__}, "
                    f"thop {type(thop_error).__name__}"
                ),
            }


def checkpoint_size_mb(path) -> float:
    """Size of a saved checkpoint on disk, in mebibytes.

    Section 31 asks for the byte convention and for equivalent artifacts to
    be compared. This reads the actual file the benchmark saved, which holds
    weights and buffers only - not optimizer state, which would carry two
    AdamW moment tensors per parameter and roughly triple the figure for
    reasons unrelated to the architecture.
    """
    return path.stat().st_size / BYTES_PER_MB if path and path.is_file() else None


@torch.no_grad()
def measure_latency(model, device, input_size: int = INPUT_SIZE,
                    batch_sizes=INFERENCE_BATCH_SIZES,
                    timed_images: int = INFERENCE_TIMED_IMAGES,
                    warmup: int = INFERENCE_WARMUP_ITERS,
                    instance: bool = False) -> dict:
    """Batch-one latency and batched throughput, measured separately.

    Section 33 requires the two distinguished. They answer different
    questions: a robot processing one frame at a time is bound by batch-one
    latency, and a cloud server scoring a queue is bound by throughput.
    Models do not rank the same way on both - a model with heavy per-layer
    launch overhead looks poor at batch one and recovers when the device is
    saturated.

    **What the timed region includes:** the forward pass and nothing else.
    Images are synthetic and already on the device, so no file reading, no
    preprocessing, no host-to-device transfer, and no mask postprocessing is
    counted. That isolates the model, which is what Section 29's efficiency
    table compares; a deployment estimate would need the rest and is not
    what this number is.
    """
    model = model.to(device).eval()
    results = {}

    for batch_size in batch_sizes:
        if instance:
            batch = [
                torch.rand(3, input_size, input_size, device=device)
                for _ in range(batch_size)
            ]
        else:
            batch = torch.randn(batch_size, 3, input_size, input_size, device=device)

        # Warm-up. The first calls on any backend pay for kernel compilation
        # and allocator growth, and on MPS the first pass through a new graph
        # is routinely an order of magnitude slower than the steady state.
        for _ in range(warmup):
            model(batch)
        synchronize(device)

        iterations = max(1, timed_images // batch_size)
        started = time.perf_counter()
        for _ in range(iterations):
            model(batch)
        synchronize(device)
        elapsed = time.perf_counter() - started

        images = iterations * batch_size
        results[batch_size] = {
            "batch_size": batch_size,
            "timed_images": images,
            "iterations": iterations,
            "total_seconds": elapsed,
            "ms_per_image": (elapsed / images) * 1000,
            "images_per_second": images / elapsed,
        }

    primary = results[batch_sizes[0]]
    throughput = results[batch_sizes[-1]]

    return {
        "latency_ms": primary["ms_per_image"],
        "latency_batch_size": primary["batch_size"],
        "throughput_images_per_second": throughput["images_per_second"],
        "throughput_batch_size": throughput["batch_size"],
        "per_batch_size": results,
        "warmup_iterations": warmup,
        "input_size": input_size,
        "timed_region": "forward pass only; excludes I/O, preprocessing, "
                        "host-device transfer and mask postprocessing",
        "synchronized": True,
        "gradients": "disabled (torch.no_grad, model.eval)",
    }


@torch.no_grad()
def measure_memory(model, device, input_size: int = INPUT_SIZE,
                   batch_size: int = 1, instance: bool = False) -> dict:
    """Device memory held during a forward pass at a given batch size.

    Section 54 asks how resolution affects memory, which needs this callable
    at more than one resolution, so the resolution is a parameter rather
    than a constant.
    """
    model = model.to(device).eval()
    reset_memory_stats(device)
    synchronize(device)

    if instance:
        batch = [
            torch.rand(3, input_size, input_size, device=device)
            for _ in range(batch_size)
        ]
    else:
        batch = torch.randn(batch_size, 3, input_size, input_size, device=device)

    model(batch)
    synchronize(device)
    used = memory_allocated_mb(device)

    del batch
    reset_memory_stats(device)

    return {
        "inference_memory_mb": used,
        "memory_batch_size": batch_size,
        "memory_input_size": input_size,
    }


def profile_model(model, model_name: str, device, checkpoint_path=None,
                  input_size: int = INPUT_SIZE, instance: bool = False,
                  measure_complexity: bool = True) -> dict:
    """Everything Sections 30 to 33 ask for, for one model, in one record.

    Complexity is computed on CPU because fvcore traces the model, and
    tracing on MPS is both slower and prone to falling back to the CPU
    implementation for individual operators anyway. The model is returned to
    ``device`` afterwards.
    """
    from .models import count_parameters, weight_size_mb

    record = {"model": model_name}
    record.update(count_parameters(model))
    record["weight_size_mb"] = round(weight_size_mb(model), 3)

    saved = checkpoint_size_mb(checkpoint_path)
    record["checkpoint_size_mb"] = round(saved, 3) if saved else None
    record["size_convention"] = (
        "MB = 1024*1024 bytes; weights and buffers at float32, "
        "excluding optimizer state"
    )

    if measure_complexity and not instance:
        # Detection models take a list of tensors and return different types
        # by mode, which fvcore's tracer cannot follow. Reported as not
        # measured rather than as a wrong number.
        record.update(complexity(model, input_size))
    else:
        record.update({
            "gmacs": None,
            "complexity_tool": None,
            "unsupported_operators": [],
            "complexity_note": (
                "N/A: detection models take a list of tensors and return a "
                "different type per mode, which the tracers cannot follow"
                if instance else "not measured"
            ),
        })

    model = model.to(device)
    record.update(measure_latency(model, device, input_size, instance=instance))
    record.update(measure_memory(model, device, input_size, instance=instance))
    record["environment"] = environment()
    return record


def measure_resolution_scaling(model, device, model_name: str,
                               resolutions=(256, 384, 512),
                               instance: bool = False) -> dict:
    """How memory and latency scale with input resolution.

    Section 54 asks two questions this answers directly - how resolution
    affected segmentation quality, and how it affected GPU memory. Quality
    needs a retrain at each resolution and is handled by the benchmark;
    memory and speed do not, and are measured here in seconds.
    """
    rows = []
    for resolution in resolutions:
        try:
            memory = measure_memory(
                model, device, input_size=resolution, instance=instance
            )
            latency = measure_latency(
                model, device, input_size=resolution,
                batch_sizes=(1,), timed_images=20, warmup=3, instance=instance,
            )
            rows.append({
                "model": model_name,
                "input_size": resolution,
                "pixels": resolution * resolution,
                "inference_memory_mb": memory["inference_memory_mb"],
                "latency_ms": latency["latency_ms"],
            })
        except RuntimeError as error:
            # Out of memory at a resolution is a finding, not a failure.
            rows.append({
                "model": model_name,
                "input_size": resolution,
                "pixels": resolution * resolution,
                "inference_memory_mb": None,
                "latency_ms": None,
                "error": str(error)[:200],
            })
    return {"model": model_name, "resolution_scaling": rows}


def save_json(payload, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
