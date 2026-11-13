# Regional inpainting design and operation

## First implementation scope

This stage implements user-painted masks for SD 1.5 only. It is intentionally a manual mask first; a prompt-driven segmentation model is not yet connected. Manual masks avoid introducing an unlicensed/large segmentation model and make the user explicitly choose which pixels may change.

## Request path

1. The browser receives the saved image asset ID from the image-generation response.
2. The user paints over the target area on a canvas. The visible red stroke is mirrored into a binary black/white PNG mask (white means repaint, black means preserve).
3. The browser sends the mask, prompt, steps, strength, seed and selected compute route to the Flask endpoint.
4. Flask retrieves the parent image bytes from PostgreSQL using the asset ID. The source image is not accepted from the browser.
5. Flask verifies the asset is an SD 1.5 image, checks the parent identity profile, validates the mask dimensions and coverage, and obtains a GPU.
6. For a cloud route, Flask forwards the source image and mask to the authenticated GPU worker. For a local route, it runs the same inpainting engine locally.
7. The engine creates a padded crop around the mask, bounds the inference crop to a maximum 768-pixel edge (with a 512-pixel minimum long edge), and uses a dedicated Diffusers StableDiffusionInpaintPipeline.
8. The identity LoRA is loaded when the parent asset has a face profile. Failure to load the LoRA aborts the repair.
9. The generated crop is resized back to its original crop size. A dilated and feathered mask blends the generated crop over the source crop. Outside that composite mask, pixels are copied directly from the original.
10. Flask stores the final image as a new generated asset, retaining the parent ID, selected identity, compute backend and mask/inference metadata.

## Configuration

    INPAINT_MODEL_ID=stable-diffusion-v1-5/stable-diffusion-inpainting
    INPAINT_MODEL_PATH=
    INPAINT_CROP_PADDING_PX=128
    INPAINT_MASK_GROW_PX=8
    INPAINT_MASK_BLUR_PX=2

The model ID is the dedicated Stable Diffusion 1.5 inpainting checkpoint. An optional local model path must contain a Diffusers model_index.json. Configure it on both the app host and GPU worker. The first use may download the model.

## Guardrails

- Only SD 1.5 parent assets are accepted. FLUX.2 Klein and SDXL are not routed through the SD 1.5 inpaint pipeline.
- A CUDA local or cloud worker is required. CPU repair is rejected.
- The mask must match source dimensions, be non-empty, and cover no more than 65% of the image.
- The source image is retrieved from the saved asset; the browser submits only the mask and correction prompt.
- The original saved asset is never edited/deleted.
- The parent face LoRA is required when a trained profile is attached. A failed adapter load stops the job.
- The cloud worker uses the same INPAINT_MODEL_ID/INPAINT_MODEL_PATH settings and shared authenticated worker route.
- The result is always a new asset with generation_mode=regional_inpaint.

## Preservation limits

The final composite directly preserves pixels outside the dilated/feathered mask. The boundary region deliberately expands beyond the user's brush to reduce seams. This means a narrow band around the selected area may change. Inpainting can still produce an incorrect repair inside the mask; visual validation is advisory and human inspection remains necessary.

The model processes a context crop, not a mathematically isolated object. It may infer content that was occluded or outside the crop poorly. Do not mark training photos as approved solely because an inpaint operation completed.

## Next stage

Add optional prompt-driven object/part segmentation, for example a locally hosted text-grounded detector plus segmentation model. It should render the proposed mask for user confirmation before any edit. Then consider automatic localized retry only after the mask is approved and before/after checks verify that unmasked regions did not drift. The segmentation weights, license, memory needs and quality threshold must be explicitly selected and configured rather than silently downloaded.
