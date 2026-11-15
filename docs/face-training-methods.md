# Face training methods

The face-training page offers two methods for Stable Diffusion 1.5 face profiles.

## LoRA (recommended default)

- Trains a compact identity adapter instead of changing the base model.
- Supports 100–3,000 training steps.
- Uses the existing LoRA training route and small adapter artifact.
- Best starting point for most face profiles, especially when you want to keep several identities separate.

## Full U-Net fine-tuning

- Updates the complete Stable Diffusion 1.5 U-Net denoiser weights for the selected profile.
- Freezes the VAE and CLIP text encoder. This is full U-Net fine-tuning, not full-pipeline/text-encoder fine-tuning.
- Supports 100–2,000 steps; the UI switches the default to 400 steps.
- Requires a CUDA GPU with at least 24 GB VRAM. The API and GHive worker check actual VRAM before starting.
- Saves to `models/face_finetunes/<profile_id>`; it never overwrites the base SD 1.5 model or the LoRA adapter.

Use this option only when the compact LoRA adapter is not giving the result you need. Full U-Net updates can overfit small datasets more easily, use substantially more VRAM, and create a much larger checkpoint. Keep the initial run around 300–400 steps and review generation quality before increasing it.

## GHive / Google Drive persistence

The GHive Drive worker exports:

```text
FACE_FINETUNE_DRIVE_REMOTE=gdrive:Local Ai/models/face-finetunes
```

After a successful full fine-tune, the worker copies that profile's U-Net folder to:

```text
gdrive:Local Ai/models/face-finetunes/<profile_id>
```

When a profile is selected on the GHive worker and the local checkpoint cache is missing, it lazily restores the folder with `rclone`. The web app does not proxy multi-gigabyte U-Net checkpoints through Render or store them in the LoRA PostgreSQL backup table. A Drive sync warning means the checkpoint trained on the current worker but was not confirmed in Drive; do not stop the rented GPU until you have repaired the sync or copied the checkpoint elsewhere.

Set the same Drive remote in the GHive rclone configuration. The remote name and folder can be changed with `FACE_FINETUNE_DRIVE_REMOTE` in the worker environment.

## Compatibility

The current face trainer is SD 1.5-specific. SDXL and FLUX.2 Klein do not use this face-training selection. LoRA and full U-Net fine-tuning modes are tracked separately for local and GHive backends, and image generation loads the active artifact type for the selected backend.
