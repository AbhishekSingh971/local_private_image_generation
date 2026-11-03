"""Train a local Stable Diffusion 1.5 face-identity LoRA on a CUDA-enabled NVIDIA GPU."""
import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer

from face_profiles import GLOBAL_BODY_PROFILE_ID, get_profile
from dataset_policy import infer_photo_kind_from_caption, select_training_files
from model_integrity import validate_sd15_model

ROOT = Path(__file__).resolve().parent
MODEL_PATH_OVERRIDE = os.environ.get("SD_MODEL_PATH", "").strip()
BASE_MODEL = Path(MODEL_PATH_OVERRIDE).expanduser() if MODEL_PATH_OVERRIDE else ROOT / "models" / "stable-diffusion-v1-5"
if not BASE_MODEL.is_absolute():
    BASE_MODEL = (ROOT / BASE_MODEL).resolve()
LEGACY_DATASET_DIR = ROOT / "data" / "face_dataset"
PROFILE_DATASET_ROOT = ROOT / "data" / "face_profiles"
OUTPUT_ROOT = ROOT / "models" / "face_loras"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class FaceDataset(Dataset):
    def __init__(self, files, tokenizer, token, class_name, resolution, description=""):
        self.files = files
        self.tokenizer = tokenizer
        self.prompt = f"a photo of {token} {class_name}"
        words = (description or "").split()
        # SD 1.5's CLIP text encoder has a fixed context window. Use short
        # rotating chunks so longer descriptions are learned over training steps
        # instead of silently truncating everything after the first chunk.
        self.description_chunks = [
            " ".join(words[start:start + 18])
            for start in range(0, len(words), 18)
        ] or [""]
        self.resolution = resolution

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image = ImageOps.fit(
                    image, (self.resolution, self.resolution),
                    method=Image.Resampling.LANCZOS,
                )
                pixels = np.asarray(image, dtype=np.float32)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot read training image: {path}") from exc
        pixels = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
        pixels = pixels / 127.5 - 1.0
        description_chunk = random.choice(self.description_chunks)
        caption_path = path.with_name(path.name + ".caption.txt")
        try:
            photo_caption = caption_path.read_text(encoding="utf-8").strip() if caption_path.is_file() else ""
        except OSError:
            photo_caption = ""
        training_prompt = self.prompt
        if photo_caption:
            training_prompt += ", " + photo_caption[:1200]
        if description_chunk:
            training_prompt += ", " + description_chunk
        input_ids = self.tokenizer(
            training_prompt, padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": pixels, "input_ids": input_ids}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a reusable face LoRA from data/face_dataset using the local SD 1.5 model."
    )
    parser.add_argument("--profile-id", default=None, help="Persistent face profile ID; uses that profile's dataset.")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--token", default="sks")
    parser.add_argument("--class-name", default="person")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--resolution", type=int, choices=(384, 512), default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dataset-purpose", choices=("identity", "body", "expression", "angle"), default=None,
                        help="Training data purpose; defaults to body for the reserved shared-body profile, identity otherwise.")
    args = parser.parse_args()
    if args.profile_id is not None and not re.fullmatch(r"[a-f0-9]{32}", args.profile_id):
        parser.error("--profile-id must be a 32-character face profile ID.")
    if args.output_name is None:
        args.output_name = args.profile_id or "my_face"
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.output_name):
        parser.error("--output-name may contain only letters, numbers, underscores and hyphens.")
    if args.profile_id and args.output_name != args.profile_id:
        parser.error("For a named face profile, --output-name must match --profile-id.")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", args.token):
        parser.error("--token may contain only letters, numbers, underscores and hyphens.")
    if args.steps < 1 or args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("Steps, batch size and gradient accumulation must be positive.")
    if args.learning_rate <= 0 or args.rank < 1:
        parser.error("Learning rate and LoRA rank must be positive.")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was not detected. Face LoRA training is not practical on this CPU-only setup. "
            "Use an NVIDIA GPU and a CUDA-enabled PyTorch installation."
        )
    try:
        validate_sd15_model(BASE_MODEL)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    dataset_dir = (
        PROFILE_DATASET_ROOT / args.profile_id / "dataset"
        if args.profile_id else LEGACY_DATASET_DIR
    )
    candidate_files = sorted(
        p for p in dataset_dir.glob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ) if dataset_dir.is_dir() else []
    purpose = args.dataset_purpose or (
        "body" if args.profile_id == GLOBAL_BODY_PROFILE_ID else "identity"
    )
    metadata_by_filename = {}
    if args.profile_id and os.environ.get("DATABASE_URL", "").strip():
        try:
            from studio_db import list_face_photo_training_metadata
            metadata_by_filename = {
                item["filename"]: item
                for item in list_face_photo_training_metadata(args.profile_id)
            }
        except Exception as exc:
            raise SystemExit(
                "Could not load dataset-purpose metadata from PostgreSQL; refusing to train on an unfiltered dataset. "
                f"Check DATABASE_URL and database connectivity. Details: {exc}"
            ) from exc
    # Read local sidecars as a portable fallback when the GHive worker is offline
    # from PostgreSQL. Database records take precedence over their local mirrors.
    if args.profile_id:
        for image_path in candidate_files:
            if image_path.name in metadata_by_filename:
                continue
            sidecar = image_path.with_name(image_path.name + ".training.json")
            if sidecar.is_file():
                try:
                    record = json.loads(sidecar.read_text(encoding="utf-8"))
                    if isinstance(record, dict) and record.get("version") == 1:
                        metadata_by_filename[image_path.name] = record
                        continue
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            caption_path = image_path.with_name(image_path.name + ".caption.txt")
            try:
                inferred_kind = infer_photo_kind_from_caption(
                    caption_path.read_text(encoding="utf-8") if caption_path.is_file() else ""
                )
            except OSError:
                inferred_kind = None
            if inferred_kind:
                metadata_by_filename[image_path.name] = {
                    "photo_kind": inferred_kind, "validation_status": "not_checked"
                }
    files, excluded_files = select_training_files(
        candidate_files, metadata_by_filename, purpose,
        include_unclassified_identity=(purpose == "identity"),
    )
    if len(files) < 5:
        raise SystemExit(
            f"Found {len(files)} eligible {purpose} photos (excluded {len(excluded_files)}) in {dataset_dir}. "
            "Add at least 5 correctly labelled, reviewed photos; 15–30 varied real references are a better starting point."
        )
    print(f"Dataset purpose: {purpose}; selected {len(files)} photos; excluded {len(excluded_files)}.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    print(f"Training from {len(files)} local photos. No remote inference API is used.")
    tokenizer = CLIPTokenizer.from_pretrained(str(BASE_MODEL), subfolder="tokenizer", local_files_only=True)
    scheduler = DDPMScheduler.from_pretrained(str(BASE_MODEL), subfolder="scheduler", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(
        str(BASE_MODEL), subfolder="text_encoder", torch_dtype=dtype,
        local_files_only=True, use_safetensors=True,
    )
    vae = AutoencoderKL.from_pretrained(
        str(BASE_MODEL), subfolder="vae", torch_dtype=dtype,
        local_files_only=True, use_safetensors=True,
    )
    unet = UNet2DConditionModel.from_pretrained(
        str(BASE_MODEL), subfolder="unet", torch_dtype=dtype,
        local_files_only=True, use_safetensors=True,
    )
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    ))
    # Keep trainable LoRA parameters in fp32 to stabilize updates.
    cast_training_params([unet], dtype=torch.float32)
    unet.enable_gradient_checkpointing()
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    unet.to(accelerator.device)

    profile = get_profile(args.profile_id) if args.profile_id else None
    description = (profile or {}).get("description", "")
    dataset = FaceDataset(files, tokenizer, args.token, args.class_name, args.resolution, description)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    trainable = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8
    )
    lr_scheduler = get_scheduler(
        "constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=args.steps
    )
    unet, optimizer, loader, lr_scheduler = accelerator.prepare(unet, optimizer, loader, lr_scheduler)
    optimizer.zero_grad(set_to_none=True)
    step = 0

    while step < args.steps:
        for batch in loader:
            with accelerator.accumulate(unet):
                pixels = batch["pixel_values"].to(dtype=dtype)
                with torch.no_grad():
                    latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
                    embeddings = text_encoder(batch["input_ids"])[0]
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (latents.shape[0],),
                    device=latents.device,
                ).long()
                noisy = scheduler.add_noise(latents, noise, timesteps)
                with accelerator.autocast():
                    prediction = unet(noisy, timesteps, embeddings).sample
                    prediction_type = scheduler.config.prediction_type
                    if prediction_type == "epsilon":
                        target = noise
                    elif prediction_type == "v_prediction":
                        target = scheduler.get_velocity(latents, noise, timesteps)
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
                if step == 1 or step % 25 == 0 or step == args.steps:
                    print(f"step {step}/{args.steps}, loss={loss.detach().item():.5f}")
            if step >= args.steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        component_output = (
            args.profile_id
            and (
                purpose in {"expression", "angle"}
                or (purpose == "body" and args.profile_id != GLOBAL_BODY_PROFILE_ID)
            )
        )
        if component_output:
            # Non-identity components must never overwrite the identity LoRA.
            output = OUTPUT_ROOT / args.profile_id / "components" / purpose
        else:
            output = OUTPUT_ROOT / args.output_name
        output.mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": 1,
            "profile_id": args.profile_id,
            "purpose": purpose,
            "selected_files": [path.name for path in files],
            "excluded_files": excluded_files,
        }
        (output / "training_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        unet = accelerator.unwrap_model(unet)
        state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=str(output), unet_lora_layers=state_dict, safe_serialization=True
        )
        print(f"Saved face LoRA: {output}")
        print(f"Use trigger token in prompts: {args.token}")
        print("Refresh the web app's face-model list to use this identity.")
    accelerator.end_training()


if __name__ == "__main__":
    main()
