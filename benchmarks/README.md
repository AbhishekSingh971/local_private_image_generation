# Private image-quality benchmark protocol

This directory contains benchmark instructions only. **Do not commit face photos, face embeddings, generated face images, LoRA weights, or other private identity data to GitHub.**

## Dataset rules

- Use only images of the profile owner or people who have explicitly consented to this use.
- Keep source photos, reference crops, generated outputs and identity embeddings in encrypted private storage on the benchmark machine.
- Assign pseudonymous profile IDs in benchmark reports. Never use a person's real name, email, or account ID.
- Keep a fixed reference set per profile: front view, left/right three-quarter views, left/right profiles, neutral expression, smile, and varied lighting where available.
- Record whether each reference is a real source image or a generated training augmentation. Do not mix those categories in evaluation.

## Fixed evaluation matrix

For every model or adapter release, run the same prompt set and seeds:

1. Identity: front, three-quarter, left profile and right profile.
2. Expression: neutral, smile, laugh and serious.
3. Scene: indoor studio, outdoor daylight and low-light environment.
4. Composition: headshot, waist-up and full-body.
5. Style: photorealism, illustration and 3D CGI.
6. Reference editing: preserve identity while changing only the requested background, clothing, pose or lighting.

Do not force the same prompt wording onto models with different supported controls. Save the original prompt, model-compiled prompt, model family, adapter versions, seed, dimensions, steps and inference settings for every result.

## Metrics to record

- Identity similarity: use a documented, licensed face-embedding evaluator on consented reference/output pairs. Record the evaluator and model version; similarity is not proof of identity.
- Face detection and landmark confidence.
- Pose/expression adherence.
- Anatomy and hand defects.
- Prompt adherence and requested-style correctness.
- Human review: identity consistency, realism, anatomy, style and obvious artifacts.
- Latency, peak VRAM, peak host RAM and generation failures.

Use both automated screening and human review. Do not set arbitrary release thresholds before measuring the current baseline; establish thresholds from the initial benchmark set and keep them fixed for subsequent comparisons.

## Release comparison rules

- Compare the same prompt, seed, dimensions and reference image whenever the model supports the same controls.
- Compare each model family against its own baseline. SD 1.5 LoRAs are not portable to SDXL or FLUX.2.
- Keep model weights, private datasets and generated benchmark media outside Git.
- A failed identity or safety/privacy review blocks release even if the average image score improves.
- Store benchmark reports with model ID, Diffusers version, adapter hashes, prompt-compiler version and GPU hardware details.

## First acceptance run

On the target CUDA worker, run the GPU smoke benchmark documented in the root README:

```bash
python gpu_acceptance.py --model-family flux2_klein --seed 12345
python gpu_acceptance.py --model-family sd15 --seed 12345
```

Keep the resulting JSON reports and images private. The script verifies that a model can load and produce an image; it does not by itself certify identity consistency or production quality.
