# Local Visual Scene Validation

## Purpose

Provide a second-pass visual review of saved image-generation output. This is separate from the deterministic technical checks (resolution, sharpness and exposure), face-geometry landmarks, and optional ArcFace identity score.

## Data flow

1. The app saves/renders the final image through the existing pipeline.
2. The app passes the final image, original user prompt, optional reference image and selected-identity flag to scene_validation.evaluate_scene_integrity.
3. When configured, the module resizes images and calls the local Ollama /api/generate endpoint.
4. The response is normalized into a bounded report and stored under generation_metadata.quality.scene_integrity.
5. The image result page displays the status, possible issues and any targeted correction hint.

No new model is downloaded by the app. The local vision model must be installed and started separately.

## Configuration

    SCENE_VISION_MODEL=your-local-vision-model-tag
    SCENE_VISION_URL=http://127.0.0.1:11434/api/generate
    SCENE_VISION_TIMEOUT_SECONDS=40

Only loopback HTTP endpoints are accepted. A LAN address or public URL is rejected, even if it is controlled by the user. This keeps the privacy boundary explicit. The model must support image inputs using Ollama's images field and JSON output mode.

## Output contract

- passed: model says the requested changes were applied, scene integrity passed, it reported no issues or unintended changes, and its self-reported confidence is at least 0.75.
- needs_review: uncertain, low-confidence, failed, contradictory or issue-bearing model report.
- not_configured: no model tag has been set.
- unavailable: local service could not be reached or returned an invalid response.

A pass is never a guarantee. The model's confidence is self-reported and is not statistically calibrated by this code.

## Retry recommendation

A correction hint may be returned as targeted_retry_hint with recommendation targeted_retry_suggested. The current implementation does not automatically rerender, overwrite the saved image, or call inpainting. A whole-image retry could damage unrelated parts of the image, so this stage deliberately leaves the decision with the user.

True selective regeneration requires a reliable localized mask (for example, an object/part segmentation mask), an SD 1.5-compatible inpainting path, preservation conditioning outside the mask, and before/after validation. Those mechanisms are not represented as complete in this change.

## Limitations

- VLMs can miss subtle defects or report false positives.
- Fine details such as exact clock-hand attachment, text, fingers or mechanical joints may be too small to assess reliably.
- ArcFace identity similarity remains a separate optional check.
- Body landmarks and head-pose checks do not reconstruct the person's exact 3D body shape.
- No validation result changes LoRA weights, sampling parameters, model selection or the saved output bytes.
