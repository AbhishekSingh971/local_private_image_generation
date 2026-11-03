# Implementation Plan v2 — Reinforced Prompt Intelligence and Identity-Aware Training

**Status:** Phase 0 audit complete; Phase 1 safeguards and reproducibility improvements implemented on this branch; CI verification pending; advanced conditioning remains staged  
**Base branch:** `feature/unified-prompt-reinforcement-training`  
**Working branch:** `feature/identity-expression-body-training-v2`

## 1. Objective

Extend the existing Private Face Studio without replacing its working Stable Diffusion, face-profile library, LoRA training, PostgreSQL persistence, background-job, or GHive worker architecture.

The milestone has two coordinated workstreams:

1. **Reinforced Prompt Intelligence:** use structured prompt enhancement plus measured generation outcomes and explicit user feedback to improve selection of prompt strategies and generation settings.
2. **Four independent model capabilities:** face identity, face angles, facial expressions, and body type. Keep their datasets, metadata, training artifacts, and evaluations separate, while composing them through a shared profile-aware generation-conditioning layer.

This document is a plan, not a claim that the features below are already implemented.

## 2. Architecture principles

- Preserve the original user prompt and intent. Enhancement may clarify and structure it, but must not invent subjects, identity traits, body attributes, scene elements, or actions.
- Treat LLM prompt enhancement and reinforcement/preference learning as different stages. The LLM proposes structured candidates; feedback learning ranks validated strategies/configurations.
- Start with a **contextual bandit / online preference-learning approach** over a bounded set of validated prompt strategies and settings. This is not gradient-based reinforcement learning of Stable Diffusion weights.
- Keep feedback and reward statistics partitioned by task, model family/checkpoint, and relevant conditioning context. Never transfer one person's identity feedback to another person's profile.
- Separate identity, angle, expression, and body training data and artifacts. Share profile identity and generation orchestration, not one mixed training dataset.
- Reuse the existing background-job lifecycle, persistence conventions, GPU routing, authenticated remote worker API, and model storage.
- Make every new model artifact versioned, evaluable, and rollback-capable.

## 3. Target architecture

```text
Single user prompt + task + explicit UI settings + profile context
                         |
                         v
              Prompt Intelligence Engine
          (LLM provider, schema validation, intent guard)
                         |
             validated prompt candidates
                         |
                         v
       Contextual Preference / Reward Selector
      (task/model/profile context; bounded strategy arms)
                         |
                         v
              Task-specific adapters
       image | video | face training | angle | expression
                         |
                         v
                Generation composer
       +-----------------+------------------+
       |                 |                  |
  Face identity     Face angle         Expression
  profile/LoRA      geometry/control   blendshape/control
       |                 |                  |
       +-----------------+------------------+
                         |
                    Body profile
              shape parameters + pose control
                         |
                         v
          SD checkpoint + compatible adapters
              ControlNet / IP-Adapter as needed
                         |
                         v
               CPU / local CUDA / GHive
                         |
                         v
         Technical checks + human preference feedback
                         |
                         v
          PostgreSQL observations + reward update
                         |
                         +----> next prompt/config selection
```

## 4. Workstream A — Reinforced Prompt Intelligence

### A1. Prompt contract and task context
- Retain one natural-language prompt field in each user-facing workflow.
- Define strict task schemas for image, video, face identity training, face-angle generation, expression generation, and body dataset/training.
- Pass explicit context separately: selected profile, task target, source-image metadata, selected model/checkpoint, backend capability, output size, and user-selected settings.
- Keep the original prompt, enhanced positive prompt, negative prompt, structured parameters, provider/model identifier, and prompt-engine version.
- Validate provider output against typed schemas and task-specific allowlists before it reaches a generation/training adapter.

### A2. Candidate strategies
- Define a small, versioned set of prompt strategies (for example: concise photorealistic, identity-preserving portrait, angle-controlled portrait, expression-controlled portrait, full-body composition, and video-motion-aware).
- Generate a bounded number of candidates; do not launch unbounded prompt expansion or repeated inference loops.
- Keep model-specific token/context limits and checkpoint compatibility explicit.
- Preserve user-specified details and negative constraints. Never silently remove constraints to improve a reward score.

### A3. Feedback and reward
- Capture explicit keep/reject/rating feedback and optional dimension-level feedback. Existing feedback endpoints and PostgreSQL tables are present, but the current selector is not yet used by generation prompts.
- Keep dimensions separate: prompt adherence, identity consistency, requested angle/expression/pose accuracy, body consistency, technical image quality, temporal video consistency, and user preference.
- Normalize each feedback dimension to a documented range and use versioned weights. Do not collapse all dimensions into a single aesthetic score.
- A candidate with a high aesthetic score but poor prompt adherence or identity consistency must not be treated as a success.
- Store the selected arm/configuration, context key, feedback provenance, reward components, aggregate reward, model/checkpoint hash, seed, scheduler, CFG, steps, resolution, controls, backend, and software versions.
- Persist observations and bandit state in PostgreSQL with migrations, idempotency, and safe concurrent updates. Audit finding: current observation and bandit-state writes are separate, and current context keys omit profile ID; fix both before using rewards to influence future requests.
- Start with UCB1 or another documented contextual-bandit baseline over prevalidated arms. Add exploration limits, minimum sample counts, cold-start defaults, and a deterministic fallback.
- Use only feedback from completed, attributable jobs. Ignore failed/cancelled jobs and incomplete evaluations.
- Add offline replay/A-B evaluation and a way to disable learning or reset a context's learned statistics.

### A4. Prompt safety and reliability
- Provider timeout, malformed JSON, schema mismatch, or provider outage must follow an explicit fallback policy. The UI must disclose when semantic enhancement was unavailable.
- Do not silently claim an unenhanced prompt was reinforced.
- Keep provider keys and endpoints in private environment configuration; never commit credentials.
- Treat prompt text, provider output, image metadata, and user feedback as untrusted input.
- Do not allow reward feedback to bypass content, privacy, consent, model-capability, or resource validation.

## 5. Workstream B — Four separate training capabilities

### B1. Face identity
- Preserve the current per-profile face LoRA workflow as the identity foundation. Audit finding: the existing trainer currently enumerates all profile-folder images, so add dataset-purpose filtering before training to prevent angle/expression samples leaking into identity training by default.
- Improve dataset review: duplicate detection, blur/occlusion checks, face crop quality, identity consistency checks, and explicit user approval.
- Track source images separately from generated/synthetic augmentation.
- Evaluate identity on held-out real reference images and varied prompts.
- Keep each profile's adapter and metadata isolated.

### B2. Face angles
- Preserve the existing nine-target angle job and its background progress/cancel/retry behavior.
- Store target yaw, pitch, roll, estimated actual pose, confidence, source image, and review status.
- Add coverage reports, duplicate/near-duplicate detection, visibility/occlusion checks, and angle-specific identity comparison.
- Prefer real multi-angle reference photos. Mark generated views as synthetic and require review before treating them as identity evidence.
- Keep angle conditioning (geometry/pose controls) separate from identity weights. Evaluate frontal, three-quarter, and profile views independently.

### B3. Facial expressions
- Preserve the existing ten-target expression-generation job.
- Add MediaPipe Face Landmarker extraction where available, including landmarks and blendshape scores; evaluate FLAME integration only after checking its license, compatibility, and resource needs.
- Store expression label/intensity, landmarks, blendshape parameters, head pose, source image, confidence, and review status.
- Keep expression deformation/conditioning separate from identity. Expression changes must not be used as identity-training labels unless explicitly approved for that purpose.
- Evaluate expression accuracy and identity preservation independently, including eyes, mouth, jaw, and cheek deformation.

### B4. Body type
- Extend the existing reserved shared global body profile/LoRA and Body Type dataset workflow; do not create a duplicate global body model. The current feature is a shared caption/prompt-driven body LoRA, not a 3D body scanner. Add an explicit distinction between shared body-style learning and per-person body-shape preservation.
- Define a body profile associated with a face profile but independently versioned.
- Capture front, back, side, and three-quarter references where available; record camera/perspective limitations and image provenance.
- Separate body shape/proportions from pose and movement. Pose changes must not rewrite body-shape identity.
- Evaluate a body reconstruction/parameter representation such as SMPL-X or a compatible alternative only after verifying license, commercial-use rights, model access, and SD checkpoint compatibility. SMPL-X is not automatically cleared for commercial deployment.
- Use pose landmarks (for example, MediaPipe Pose) as structural observations, not as a substitute for full 3D body geometry.
- Store shape parameters, pose parameters, landmark confidence, reference images, adapter/checkpoint version, and review status.
- Train a body-specific adapter only if data volume, quality, and experiments show it improves consistency. Otherwise use explicit geometry/pose conditioning rather than forcing an unnecessary LoRA.

## 6. Shared generation composer

- Resolve a selected face profile and optional body profile.
- Compose identity, angle, expression, body-shape, and pose controls independently.
- Enforce compatibility between the base checkpoint, LoRAs, ControlNet, IP-Adapter, and geometry assets.
- Define precedence and bounded weights; reject incompatible or unsupported combinations instead of silently dropping controls.
- Preserve CPU inference fallback and the existing local CUDA/GHive routing.
- Use one job manifest for reproducibility, but keep each component's version and parameters individually recorded.
- Add image and video regression tests for identity drift, angle correctness, expression correctness, body consistency, and video temporal stability.

## 7. To-do list and acceptance criteria

### Phase 0 — Repository and compatibility audit
- [x] Inspect current prompt-engine call sites, reward module, core job/database models, local worker, and GHive worker. See `docs/PHASE_0_AUDIT.md`.
- [x] Map current angle/expression dataset generation and identify exact extension points. Existing nine-angle and ten-expression jobs will be extended, not recreated.
- [ ] Confirm base checkpoint and all ControlNet/IP-Adapter/model-family compatibility before selecting new model assets.
- [ ] Inventory and execute the full test suite to establish runtime baseline results before source changes. GitHub source inspection alone does not verify tests.
- **Done when:** a file-level change map and passing/failing baseline are recorded; no existing route or job behavior is changed.

### Phase 1 — Prompt intelligence and reinforcement foundation
- [ ] Define typed prompt/task/context/candidate/feedback/reward schemas.
- [ ] Add bounded prompt strategy candidates and schema validation.
- [ ] Add prompt provenance and transparent fallback status.
- [ ] Add feedback capture for completed generation jobs.
- [ ] Add PostgreSQL tables/migrations for observations, reward components, arm/configuration versions, and bandit state.
- [ ] Add task/model/profile-scoped contextual bandit with cold-start and exploration bounds.
- [ ] Add tests for reward normalization, context isolation, invalid feedback, concurrency/idempotency, provider failures, and intent preservation.
- **Done when:** a completed job can record feedback and influence a later validated strategy choice; unrelated profiles/tasks remain isolated; learning can be disabled.

### Phase 2 — Face identity and angle quality
- [ ] Audit existing face LoRA and angle generation before modifying them.
- [ ] Add dataset quality and duplicate checks with reviewable results.
- [ ] Improve pose confidence and angle coverage reports.
- [ ] Add identity comparison on held-out references.
- [ ] Add tests ensuring synthetic angle outputs are marked synthetic/Needs review until accepted.
- **Done when:** angle coverage improves without measurable regression in identity evaluation and existing background jobs still work.

### Phase 3 — Expression training
- [ ] Add expression metadata and landmark/blendshape extraction.
- [ ] Add expression dataset review and intensity labels.
- [ ] Add an independently versioned expression conditioning/adapter path.
- [ ] Add tests that expression changes do not alter the selected identity profile or angle metadata.
- **Done when:** expression controls produce measurable expression improvement while identity scores remain within an agreed regression tolerance.

### Phase 4 — Body profile and training
- [ ] Add persistent body-profile metadata associated with, but separate from, face identity.
- [ ] Add body dataset upload, grouping, angle/coverage validation, and review.
- [ ] Add body shape/pose extraction behind a capability interface.
- [ ] Complete license and model compatibility review before adding SMPL-X or another 3D model dependency.
- [ ] Prototype geometry/pose conditioning; compare against a body-specific adapter before selecting the training method.
- [ ] Add held-out body consistency and pose tests.
- **Done when:** body shape is stable across pose changes and face identity remains unchanged in combined generations.

### Phase 5 — Shared composition and generation
- [ ] Add profile-aware conditioning composer for identity, angle, expression, body shape, and pose.
- [ ] Validate model/control compatibility and reject unsupported combinations.
- [ ] Pass the same validated manifest through local and GHive worker execution.
- [ ] Persist all component versions and settings with image/video jobs.
- [ ] Add end-to-end image and video tests, including worker failure/retry and CPU fallback.
- **Done when:** the same selected profile and conditioning manifest reproduce comparable results across supported backends.

### Phase 6 — Evaluation, release, and rollback
- [ ] Add a fixed, consented evaluation set with train/validation separation.
- [ ] Compare new outputs against the previous stable pipeline.
- [ ] Add per-dimension metrics and human review; do not rely on one aesthetic score.
- [ ] Add adapter/model promotion and rollback.
- [ ] Run unit, integration, migration, and worker compatibility tests.
- [ ] Document setup, model licenses, data handling, GPU requirements, and recovery.
- **Done when:** release gates pass and rollback to previous adapters/configuration is verified.

## 8. Not-to-do list / explicit exclusions

- Do not merge this branch into `main` automatically.
- Do not overwrite or delete the existing unified prompt branch.
- Do not replace the working Stable Diffusion pipeline, face LoRA trainer, database, job system, or GHive worker as part of the initial phases.
- Do not mix face identity, angle, expression, and body datasets into one unlabeled dataset or one indiscriminate LoRA.
- Do not claim that prompt enhancement is reinforcement learning.
- Do not train diffusion weights with online RL in this milestone.
- Do not update reward statistics from failed, cancelled, unreviewed, or unattributed jobs.
- Do not optimize only for aesthetic preference or let reward override prompt adherence, identity, user constraints, or safety validation.
- Do not generate unlimited prompt variants or launch repeated inference loops to chase a reward.
- Do not silently fall back from full fine-tuning to LoRA, or from semantic prompt enhancement to a generic prompt.
- Do not treat synthetic face angles or synthetic body images as ground-truth identity evidence without review.
- Do not infer exact hidden body geometry or measurements from a single 2D photograph.
- Do not add SMPL-X or other model dependencies before license, commercial-use, compatibility, and compute review.
- Do not load all large models onto GPU simultaneously by default; use capability-aware loading and the existing job queue.
- Do not commit API keys, provider credentials, database URLs, worker tokens, personal images, private datasets, generated media, or model weights.
- Do not change unrelated UI/theme, deployment, or provider configuration.
- Do not remove Google Colab support or alter its existing setup as part of this milestone.

## 9. Data, privacy, and operational requirements

- Process only images the user owns or has permission to use.
- Keep profile datasets isolated and make source/generated provenance visible.
- Support dataset and profile deletion according to the existing data-retention contract.
- Keep secrets in private environment variables and redact them from logs.
- Make remote-worker transfer explicit; images/prompts sent to GHive are processed on that host.
- Persist job state and completed artifacts before releasing ephemeral compute resources.
- Never report a model as trained or promoted until its checkpoint is saved, reloaded, and evaluated.

## 10. Rollout strategy

1. Commit this plan and branch baseline first.
2. Complete Phase 0 audit and record baseline tests.
3. Implement prompt reinforcement as a separately testable vertical slice.
4. Implement expression and body modules behind feature flags, reusing existing angle and identity workflows.
5. Integrate the composer only after each component passes independent tests.
6. Keep all work on this feature branch until the user reviews the results.
7. Open a pull request only after tests and documentation are complete; do not merge without explicit approval.


## 11. Phase 0 audit addendum

The repository audit is recorded in `docs/PHASE_0_AUDIT.md`. It confirms that the current branch already has a structured PromptEngine, reward/evaluation and PostgreSQL feedback modules, feedback endpoints, nine angle targets, ten expression targets, a hidden shared global body LoRA, and local/cloud GPU routing. However, the PromptEngine and bandit selector are not yet connected as an active prompt-selection loop in generation. Existing identity LoRA training enumerates all image files in the profile dataset folder, so purpose-aware dataset selection is a prerequisite. The current global body feature is a shared body-type LoRA, not full 3D person-specific body reconstruction. Runtime tests and model/control compatibility checks remain open.


## 12. Phase 1 implementation record (2026-11-03)

Implemented on `feature/identity-expression-body-training-v2`:

- Added purpose-aware training selection. Identity training uses source/manual photos; the shared body profile uses only body-labelled photos. Expression and angle photos are excluded from identity training. Photos marked `needs_review` are excluded.
- Added lightweight PostgreSQL training metadata lookup and portable `.training.json` sidecars beside local dataset images. Existing unknown legacy images remain eligible only for ordinary identity profiles; they are never silently treated as body examples.
- Added a per-training-run manifest recording selected and excluded files. Experimental expression/angle LoRA outputs are placed under a separate `components/<purpose>` directory so they cannot overwrite the identity LoRA.
- Extended the local GPU image pipeline and GHive worker to apply the existing SD 1.5 face IP-Adapter when a reference image is supplied. This is GPU-only and optional; CPU generation avoids the extra image-encoder memory overhead.
- Added advisory technical image-quality results to saved asset metadata. These checks cover resolution, sharpness and exposure only; they do not certify face identity or anatomy.
- Added persistent generation metadata to saved media assets, including seed, steps, dimensions/preset and selected conditioning. Existing databases are upgraded with an additive JSONB column.
- Added video seed and playback-FPS controls, passed them through to GHive, persisted them with saved videos, and enabled attention slicing as a lower-memory option on both local and cloud AnimateDiff pipelines.
- Added profile identity and strategy-version scoping to contextual preference-learning keys so one person's feedback is not reused for another profile.
- Added regression tests for dataset isolation, video settings, profile-scoped reward contexts and asset metadata persistence.

Still staged (not claimed as complete):

- Expression/angle LoRA training is not yet exposed as a selectable UI training workflow or loaded as independent adapters during image/video inference. The CLI output isolation is only a safe foundation.
- Full-body OpenPose/Depth ControlNet conditioning is not wired. The current pose ControlNet is face/head-pose oriented; a compatible SD 1.5 full-body model and preprocessing path must be validated first.
- The shared body LoRA remains a body-type/style adapter, not a per-person 3D body identity model.
- The preference selector is not yet connected to choose prompt strategies at generation time, and feedback observation plus bandit-state writes still need one atomic/concurrency-safe transaction.
- Video still uses AnimateDiff text-to-video. It has no reference-frame/video conditioning, no body-pose control, and no temporal identity evaluator. Seed/FPS/attention changes improve control and reliability, not motion-model capability.
- No promise of zero distortion: identity, anatomy and pose must be measured and human-reviewed; a single reference image cannot define unseen geometry.


## 13. Phase 2 implementation record (2026-11-03)

Additional integration now committed on this feature branch:

- Connected PromptEngine to the actual image and video generation routes. When no semantic prompt provider is configured, the fallback is transparent and retains the user's original prompt. Provider failures are surfaced rather than silently pretending enhancement happened.
- Connected a bounded contextual bandit to a small set of reviewed prompt additions: identity preservation, full-body framing and video temporal stability. Selection context is scoped by task, model family, backend, control mode, profile and strategy version. The feedback configuration ID now matches the selected strategy arm.
- Made feedback observation insertion and bandit-state update one PostgreSQL transaction. A row lock serializes concurrent updates to the same contextual state, preventing lost updates between the observation and learned state.
- Exposed purpose-specific LoRA training in the Face Models UI for identity, angle, expression and profile-specific body adapters. The local trainer and GHive worker both pass the purpose to the dataset-filtering trainer; component jobs do not overwrite the identity-ready flag.
- Added a component-adapter status endpoint and a Sync component to GHive action. Component weights are uploaded to isolated components/<purpose> folders and do not replace the root identity LoRA.
- Image inference now loads selected angle/expression component adapters independently when their weights exist. Missing component weights fall back to prompt-only control and are reported in saved generation metadata. AnimateDiff now has the same optional angle/expression adapter selection on local CUDA and GHive.
- Added optional full-body pose conditioning using an SD 1.5 OpenPose ControlNet. The UI accepts a pre-rendered OpenPose skeleton/control map, not a normal photo. Text-to-image and reference-image img2img ControlNet paths are selected separately. Configure BODY_POSE_CONTROLNET_ID and optional BODY_POSE_CONTROLNET_SUBFOLDER to select a compatible model.
- Added advisory sampled video quality checks for blur, exposure, frame-to-frame luminance changes and optical-flow motion spikes. The report explicitly leaves face identity consistency as not_assessed and always requires human review; failed or unavailable checks never discard a generated video.

Still not implemented and intentionally not represented as complete:

- No true image-to-video or video-to-video reference-conditioned pipeline is integrated. AnimateDiff remains text-to-video.
- The OpenPose upload expects a rendered skeleton/control map. Automatic extraction of a skeleton from an ordinary photo is not yet bundled; a normal photograph should not be uploaded in this field.
- Temporal metrics are advisory. They do not compare face embeddings, verify anatomy, establish prompt adherence or guarantee identity consistency.
- The prompt strategy bandit learns only from explicit feedback recorded for saved assets. It is not diffusion-model RL and does not update model weights.
- Full fine-tuning remains blocked until a separately validated trainer is installed.