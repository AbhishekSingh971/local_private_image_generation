# GHive: load SD 1.5 from Google Drive

This path is for GHive instances where `/dev/fuse` is unavailable. Diffusers
expects local random-access files, so this is **selective on-demand caching**, not
zero-byte remote inference or a POSIX mount.

## Configure

1. Install rclone and configure a remote named `gdrive`.
2. Confirm the nested Diffusers snapshot exists:
   `rclone lsf "gdrive:Local Ai/models/stable-diffusion-v1-5/vae"`
3. Set `REMOTE_API_TOKEN` in the GHive terminal.
4. Start the worker and tunnel:

```bash
export REMOTE_API_TOKEN='YOUR_NEW_LONG_RANDOM_SECRET'
bash run_ghive_drive_worker.sh
```

Optional overrides:

- `SD_MODEL_DRIVE_REMOTE`: rclone path to the Diffusers model root. Default:
  `gdrive:Local Ai/models/stable-diffusion-v1-5/vae`.
- `SD_MODEL_CACHE_DIR`: local cache directory. Default:
  `~/.cache/local_private_image_generation/sd15`.

## What is fetched

The cache helper downloads only the 13 files needed by the current SD 1.5
Diffusers pipeline: model index, component configs, tokenizer files, and the
three standard SafeTensors weights (text encoder, UNet, VAE). It skips safety
checker weights because the app disables the safety checker for this pipeline.
It does not fetch CKPT, BIN, FP16 duplicate variants, Hugging Face files, or the
rest of the 48 GiB Drive folder.

The selected full-precision weights total approximately 4.0 GiB, plus small
configuration/tokenizer files. The files are fetched one at a time into
`.partial` paths and atomically renamed after successful transfer. A completed
cache is reused on the next worker start. Put `SD_MODEL_CACHE_DIR` on a
persistent GHive volume if one is available; otherwise a fresh GHive instance
must fetch the files again.

## Limitation

This does not stream tensor pages directly from Google Drive into VRAM. With no
FUSE and the current Diffusers loader, the required weight files must be present
as local files before `from_pretrained` can load them. The benefit is avoiding
the 48 GiB bulk copy and fetching only the exact inference files from Drive.
