"""Manual GPU acceptance test for SD 1.5 and FLUX.2 Klein.

Run this on the actual CUDA host before enabling a model for production traffic.
It writes a generated PNG and a JSON report containing hardware, latency and
peak CUDA-memory measurements. It does not use or upload face datasets.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image, ImageOps

from image_generation import resolve_image_dimensions
from model_registry import model_configuration, resolve_model_family


def parse_args():
    parser = argparse.ArgumentParser(description="Run a real CUDA image-generation acceptance test.")
    parser.add_argument("--model-family", choices=("sd15", "sdxl", "flux2_klein"), default="flux2_klein")
    parser.add_argument("--prompt", default="A detailed cinematic photograph of a red fox standing in a snowy pine forest at sunrise.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--reference-image", type=Path, default=None)
    parser.add_argument("--aspect-ratio", choices=("portrait", "square", "landscape"), default="square")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--strength", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, default=Path("data") / "gpu_acceptance")
    return parser.parse_args()


def main():
    args = parse_args()
    family = resolve_model_family(args.model_family)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("GPU acceptance failed: torch.cuda.is_available() is False.")

    if family == "flux2_klein":
        if args.negative_prompt.strip():
            raise SystemExit("FLUX.2 Klein distilled does not accept a negative prompt.")
        steps = 4 if args.steps is None else args.steps
        if steps != 4:
            raise SystemExit("FLUX.2 Klein distilled requires exactly 4 inference steps.")
    else:
        steps = 25 if args.steps is None else args.steps
        if not 5 <= steps <= 25:
            raise SystemExit("SD 1.5 and SDXL acceptance steps must be between 5 and 25.")
    if not 0.2 <= args.strength <= 0.95:
        raise SystemExit("Reference strength must be between 0.20 and 0.95.")

    width, height = resolve_image_dimensions(args.aspect_ratio, family)
    reference = None
    if args.reference_image:
        with Image.open(args.reference_image) as source:
            reference = ImageOps.exif_transpose(source).convert("RGB")
            reference.load()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{family}-{args.seed}.png"
    report_path = args.output_dir / f"{family}-{args.seed}.json"
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    prompt_token_count = None
    if family == "flux2_klein":
        from flux2_engine import generate_flux2_image

        result = generate_flux2_image(
            args.prompt,
            image=reference,
            seed=args.seed,
            width=width,
            height=height,
            source=model_configuration(family)["source"],
        )
        image = result["image"]
        prompt_token_count = result["prompt_token_count"]
    else:
        import app as local_engine
        from generation_contracts import validate_diffusion_prompt_capacity

        mode = "img2img" if reference is not None else "txt2img"
        pipeline = local_engine._load_cpu_or_gpu_pipeline(
            mode, None, "local_gpu", model_family=family
        )
        effective_reference = (
            ImageOps.fit(reference, (width, height), method=Image.Resampling.LANCZOS)
            if reference is not None else None
        )
        token_counts = validate_diffusion_prompt_capacity(
            pipeline, args.prompt, args.negative_prompt
        )
        prompt_token_count = token_counts
        kwargs = {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt or None,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": 7.5,
            "generator": torch.Generator(device="cuda").manual_seed(args.seed),
        }
        if effective_reference is not None:
            kwargs.update(image=effective_reference, strength=args.strength)
        image = pipeline(**kwargs).images[0]

    torch.cuda.synchronize()
    elapsed_seconds = round(time.perf_counter() - started, 3)
    image.save(output_path, format="PNG")
    config = model_configuration(family)
    device = torch.cuda.get_device_properties(0)
    report = {
        "status": "passed",
        "model_family": family,
        "model_id": config["model_id"],
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_total_gb": round(device.total_memory / (1024 ** 3), 2),
        "vram_peak_allocated_gb": round(torch.cuda.max_memory_allocated(0) / (1024 ** 3), 2),
        "vram_peak_reserved_gb": round(torch.cuda.max_memory_reserved(0) / (1024 ** 3), 2),
        "elapsed_seconds": elapsed_seconds,
        "steps": steps,
        "seed": args.seed,
        "requested_canvas": {"width": width, "height": height},
        "output_dimensions": {"width": image.width, "height": image.height},
        "reference_image_used": bool(reference),
        "prompt_token_count": prompt_token_count,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt if family != "flux2_klein" else "",
        "output_file": str(output_path),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"GPU acceptance passed. Image: {output_path}; report: {report_path}")


if __name__ == "__main__":
    main()
