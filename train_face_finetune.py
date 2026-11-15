"""Full U-Net fine-tuning for a private SD 1.5 face profile.

This is intentionally separate from train_face_lora.py. It updates the complete
SD 1.5 denoiser while freezing the VAE and CLIP text encoder. The base model is
never modified; output is saved into models/face_finetunes/<profile_id>.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from transformers import CLIPTextModel, CLIPTokenizer

from face_profiles import get_profile, full_finetune_model_path
from train_face_lora import EXTENSIONS, FaceDataset
from training_approval import select_approved_training_files
from training_modes import FULL_FINETUNE_MIN_VRAM_GB

ROOT = Path(__file__).resolve().parent
MODEL_PATH_OVERRIDE = os.environ.get("SD_MODEL_PATH", "").strip()
BASE_MODEL = (
    Path(MODEL_PATH_OVERRIDE).expanduser()
    if MODEL_PATH_OVERRIDE
    else ROOT / "models" / "stable-diffusion-v1-5"
)
PROFILE_DATASET_ROOT = ROOT / "data" / "face_profiles"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full SD 1.5 U-Net fine-tuning from explicitly approved face photos."
    )
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--approved-filename", action="append", default=[])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--resolution", type=int, choices=(384, 512), default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{32}", args.profile_id):
        parser.error("--profile-id must be a 32-character face profile ID.")
    if len(args.approved_filename) < 5:
        parser.error("Full fine-tuning requires at least 5 explicitly approved photos.")
    if not 100 <= args.steps <= 2000:
        parser.error("Full fine-tuning steps must be between 100 and 2000.")
    if args.batch_size != 1:
        parser.error("Full fine-tuning currently supports a batch size of 1.")
    if args.gradient_accumulation_steps < 1:
        parser.error("Gradient accumulation must be positive.")
    if args.learning_rate <= 0:
        parser.error("Learning rate must be positive.")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Full fine-tuning requires a CUDA-enabled NVIDIA GPU.")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if vram_gb < FULL_FINETUNE_MIN_VRAM_GB:
        raise SystemExit(
            f"Full U-Net fine-tuning requires at least {FULL_FINETUNE_MIN_VRAM_GB:.0f} GB VRAM; "
            f"this GPU reports {vram_gb:.1f} GB. Use LoRA or rent a larger GPU."
        )
    if not (BASE_MODEL / "model_index.json").is_file():
        raise SystemExit(
            f"Stable Diffusion 1.5 model missing at {BASE_MODEL}. "
            "Set SD_MODEL_PATH to the complete Diffusers model folder first."
        )

    profile = get_profile(args.profile_id)
    if not profile:
        raise SystemExit("Face profile not found.")
    dataset_dir = PROFILE_DATASET_ROOT / args.profile_id / "dataset"
    try:
        files = select_approved_training_files(dataset_dir, args.approved_filename, EXTENSIONS)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if len(files) < 5:
        raise SystemExit(f"Found {len(files)} approved images; at least 5 are required.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    print(
        f"Starting full SD 1.5 U-Net fine-tuning: {len(files)} approved photos, "
        f"{args.steps} steps, {args.resolution}px, {vram_gb:.1f} GB VRAM."
    )
    print("VAE and CLIP text encoder are frozen. The original base model is read-only.")

    tokenizer = CLIPTokenizer.from_pretrained(
        str(BASE_MODEL), subfolder="tokenizer", local_files_only=True
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        str(BASE_MODEL), subfolder="scheduler", local_files_only=True
    )
    text_encoder = CLIPTextModel.from_pretrained(
        str(BASE_MODEL), subfolder="text_encoder", torch_dtype=dtype, local_files_only=True
    )
    vae = AutoencoderKL.from_pretrained(
        str(BASE_MODEL), subfolder="vae", torch_dtype=dtype, local_files_only=True
    )
    unet = UNet2DConditionModel.from_pretrained(
        str(BASE_MODEL), subfolder="unet", torch_dtype=dtype, local_files_only=True
    )

    vae.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()
    unet.requires_grad_(True).train()
    unet.enable_gradient_checkpointing()
    # Full-precision trainable weights are more stable with AdamW; autocast keeps
    # the forward pass mixed precision. The VRAM preflight above prevents small GPUs
    # from silently starting a job that is unlikely to fit.
    cast_training_params([unet], dtype=torch.float32)
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    unet.to(accelerator.device)

    dataset = FaceDataset(
        files, tokenizer, profile["token"], "person", args.resolution,
        profile.get("description", ""),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    trainable = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, betas=(0.9, 0.999),
        weight_decay=1e-2, eps=1e-8, foreach=False,
    )
    lr_scheduler = get_scheduler(
        "constant", optimizer=optimizer, num_warmup_steps=0,
        num_training_steps=args.steps,
    )
    unet, optimizer, loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, loader, lr_scheduler
    )
    optimizer.zero_grad(set_to_none=True)
    step = 0

    while step < args.steps:
        for batch in loader:
            with accelerator.accumulate(unet):
                pixels = batch["pixel_values"].to(
                    device=accelerator.device, dtype=dtype, non_blocking=True
                )
                with torch.no_grad():
                    latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
                    embeddings = text_encoder(batch["input_ids"].to(accelerator.device))[0]
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                with accelerator.autocast():
                    prediction = unet(noisy_latents, timesteps, embeddings).sample
                    prediction_type = noise_scheduler.config.prediction_type
                    if prediction_type == "epsilon":
                        target = noise
                    elif prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unsupported prediction type: {prediction_type}")
                    loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                step += 1
                if step == 1 or step % 10 == 0 or step == args.steps:
                    print(f"step {step}/{args.steps}, loss={loss.detach().item():.6f}")
            if step >= args.steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output = full_finetune_model_path(args.profile_id)
        staging = output.with_name(output.name + ".tmp-" + uuid.uuid4().hex)
        backup = output.with_name(output.name + ".previous")
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)
        try:
            trained_unet = accelerator.unwrap_model(unet).to(dtype=torch.float16)
            trained_unet.save_pretrained(str(staging), safe_serialization=True)
            metadata = {
                "training_mode": "full_finetune",
                "model_family": "sd15",
                "profile_id": args.profile_id,
                "steps": args.steps,
                "resolution": args.resolution,
                "learning_rate": args.learning_rate,
                "approved_photo_count": len(files),
                "vram_gb": round(vram_gb, 2),
                "frozen_components": ["vae", "text_encoder"],
            }
            (staging / "training_mode.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            if output.exists():
                output.replace(backup)
            try:
                staging.replace(output)
            except Exception:
                if backup.exists() and not output.exists():
                    backup.replace(output)
                raise
            shutil.rmtree(backup, ignore_errors=True)
            print(f"Saved full fine-tuned SD 1.5 U-Net: {output}")
            print(f"Trigger token: {profile['token']}")
            print("The base SD 1.5 model remains unchanged.")
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    accelerator.end_training()


if __name__ == "__main__":
    main()
