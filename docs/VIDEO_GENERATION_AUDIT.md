# AnimateDiff Video Generation Audit

**Repository:** `AbhishekSingh971/local_private_image_generation`  
**Branch:** `feature/identity-expression-body-training-v2`  
**Audit date:** 2026-11-03

## Current pipeline

- Base checkpoint: the app's local Stable Diffusion 1.5 model directory.
- Motion component: the configured AnimateDiff motion adapter.
- Execution: local CUDA or authenticated GHive/cloud GPU worker. CPU video generation is explicitly rejected.
- Output: frames are encoded to MP4 at a selected playback FPS and saved to PostgreSQL plus the local video folder.
- Identity: an optional face-profile LoRA is loaded into the AnimateDiff pipeline. The video path does not currently apply the image-generation IP-Adapter reference path.
- UI limits: 8, 16 or 24 generated frames; 5–20 inference steps.
- Progress: denoising progress is reported; MP4 encoding and database persistence are separate stages.

## Findings and changes made

1. **Very short clips:** the old UI rendered at fixed 8 FPS. With 8/16/24 frames this is approximately 1/2/3 seconds. FPS is now configurable at 6, 8 or 12, and the chosen FPS is persisted. This changes playback speed/duration, not the number of motion states the model synthesizes.
2. **Hard-to-reproduce results:** the worker previously generated a time-based random seed. A user-supplied optional seed is now passed through the app and GHive worker and saved in the asset's generation metadata. A seed is reproducible only with compatible model/software versions and settings.
3. **VRAM pressure:** both local and remote AnimateDiff pipelines now enable attention slicing in addition to VAE slicing. This trades speed for lower attention-memory use; it does not increase visual fidelity.
4. **Missing experiment metadata:** seed, frame count, FPS, steps, selected face profile and motion adapter name are now stored with the saved video.

## Remaining quality limitations

- The current UI is text-to-video only: no reference image/video input, no explicit pose map, and no motion brush or region control.
- A face LoRA can improve the learned appearance but does not guarantee the same face in every frame. There is no per-frame face-embedding comparison or temporal drift score.
- There is no temporal flicker, optical-flow, motion-smoothness or frame-anatomy evaluation gate.
- The video pipeline does not currently load the shared body LoRA or independent expression adapters.
- Increasing output dimensions through FFmpeg is resizing, not true high-resolution generation.
- AnimateDiff is a short-clip model; changing FPS cannot create new motion detail or make a 24-frame clip equivalent to a longer, higher-frame-rate video.

## Recommended next stages

1. Add reference-frame conditioning only after selecting and testing a compatible image-to-video/video-to-video pipeline and its SD checkpoint/motion adapter compatibility. Do not pass an image to the current text-to-video pipeline and assume it is conditioned.
2. Add pose guidance using a compatible video/ControlNet route with validated body keypoints. The existing face-pose ControlNet is not a full-body OpenPose solution.
3. Add per-frame face detection/embedding consistency and temporal flicker/motion checks. Mark uncertain clips for review; do not auto-approve them as training data.
4. Add optional frame interpolation and upscaling as explicit post-processing stages, with clear labels. They can improve playback smoothness or pixel size but cannot repair identity drift or malformed source frames.
5. Benchmark VRAM, generation time, and quality at 8/16/24 frames on the actual GHive GPU before increasing frame count or resolution limits.

**Important:** The current phase improves repeatability, observability and memory behavior. It does not turn AnimateDiff into a reference-conditioned or identity-locked video model, and it cannot guarantee distortion-free output.

## Phase 2 update (2026-11-03)

The following improvements are now wired into the application:

- Optional angle and expression component LoRAs can be selected in the video UI and loaded separately from the identity adapter on local CUDA and GHive.
- Video prompts pass through the shared PromptEngine and a bounded temporal-stability strategy selector. The user's requested action remains in the prompt; temporal safeguards are appended as constraints.
- Saved video assets receive an advisory sampled technical report for blur, exposure, luminance flicker and motion spikes when OpenCV is available. The report always states identity_consistency: not_assessed and requires human review. It is not an identity metric or a video approval gate.
- Full-body OpenPose ControlNet is available for image generation using a supplied pre-rendered skeleton map; this does not currently condition AnimateDiff video.

## Remaining video limitations

- AnimateDiff is still text-to-video. There is no compatible image-to-video or video-to-video reference-conditioning pipeline integrated, so a reference image cannot yet anchor the same identity across frames.
- No per-frame face embedding comparison, identity-drift score, or frame-level anatomy check exists. The temporal report is a technical heuristic only.
- There is no full-body pose conditioning in the video path. Image-generation OpenPose support does not transfer automatically to AnimateDiff.
- Frame interpolation and AI super-resolution remain separate future post-processing stages. FFmpeg resizing does not create detail.
- A successful technical report does not prove a good clip. Inspect the actual frames before using them as training data.