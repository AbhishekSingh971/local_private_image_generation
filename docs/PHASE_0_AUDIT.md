# Phase 0 Repository Audit

**Repository:** `AbhishekSingh971/local_private_image_generation`  
**Audited ref:** `feature/identity-expression-body-training-v2`  
**Audit date:** 2026-11-03  
**Scope:** read-only inspection of current source and documentation. No application source code was changed during this audit.

## Executive summary

The repository already contains useful building blocks. The new milestone should extend them rather than build parallel systems.

| Area | Existing implementation observed | Gap for this milestone |
|---|---|---|
| Prompt engine | `prompt_intelligence/engine.py` accepts one prompt and returns structured positive/negative prompts and parameters. | The Flask generation routes still call `build_generation_prompts()`; the new PromptEngine is not wired into the active image/video/training request flow. |
| Preference/reward | `reward.py` provides UCB1 arm statistics; `evaluation.py` combines human and limited heuristic scores; feedback routes exist for saved assets and face photos. | Feedback currently records the selected configuration and updates stored bandit state, but the generation path does not select its next prompt strategy from that learned state. |
| Feedback persistence | `feedback_store.py` stores reward observations and bandit JSON in PostgreSQL. | Profile identity is not part of `PreferenceBandit.context_key()`; different profiles can share statistics when task/model/backend/control are otherwise equal. The feedback row and bandit-state update are separate database operations, so concurrent updates can lose a bandit-state increment. |
| Face angles | `training_generation.py` defines nine angle targets; `app.py` schedules durable background jobs and `pose_validation.py` estimates head pose and marks uncertain results for review. | Improve coverage/identity evaluation and preserve synthetic provenance. Do not recreate the existing angle job system. |
| Facial expressions | `training_generation.py` defines ten expression targets; the app generates expression variations separately from angle targets. | Add expression landmark/blendshape metadata and a genuinely independent expression-training/evaluation path. Current generated expression photos reside in the same per-profile local dataset folder as other face photos. |
| Face identity LoRA | `train_face_lora.py` reads all images under the selected profile's dataset folder. | Dataset-purpose filtering is required so angle/expression augmentation does not automatically enter the identity LoRA training set. |
| Body type | A reserved shared profile (`GLOBAL_BODY_PROFILE_ID`) and global body LoRA path exist; body-type groups and generation/editing paths are present. | This is a prompt/caption-driven global body LoRA workflow, not a 3D body scanner or independent per-person body-geometry model. Extend this feature rather than creating a duplicate global body model. Add a clear distinction between shared body-style learning and a person's body-profile preservation. |
| Model composition | Local pipeline can load a face identity adapter and the global body adapter; body adapter is currently assigned a fixed weight in the inspected local pipeline. | Add explicit, validated composition for identity, angle, expression, body shape and pose, with independently versioned components and compatibility checks. |
| Compute | Existing CPU/local CUDA/cloud worker routing and authenticated worker endpoints are present. | Preserve the worker contract; send a validated, versioned conditioning manifest through local and GHive execution. |

## Important source observations

1. `app.py` imports `build_generation_prompts`, feedback helpers, body-profile helpers, pose validation, and target manifests. The inspected generation paths use the legacy prompt builder. No `PromptEngine` call was found in the inspected Flask application source.
2. `/api/assets/<asset_id>/feedback` and `/api/face-photos/<photo_id>/feedback` accept explicit ratings and call `record_generation_feedback()`.
3. `record_generation_feedback()` evaluates the feedback, loads the contextual bandit, calls `record()`, saves the observation, then saves bandit state. The selector's `select()` method is not currently used in the inspected generation prompt path.
4. The current bandit context key uses task, model family, backend class and control mode, but not profile ID or strategy version. Add profile scoping where identity-sensitive learning is used; keep a separate profile-independent scope only for genuinely global prompt preferences.
5. Feedback storage lazily creates its own PostgreSQL tables. Feedback insertion and bandit-state persistence are separate transactions. Before relying on online updates, make the observation and state update atomic or use a concurrency-safe aggregate update/locking strategy.
6. The identity training script enumerates all image files in `data/face_profiles/<profile-id>/dataset/`. Dataset groups carry types such as face_angles, expressions and body_type in database metadata, but the trainer's file enumeration does not itself filter by purpose. This is a material cross-contamination risk for identity LoRA training.
7. The global body profile is a real existing feature. It uses a separate reserved profile/token and a separately loaded LoRA adapter. It should not be described as complete 3D body reconstruction; no SMPL-X/FLAME-based body fitting pipeline was established by this audit.

## Updated implementation decisions

- Keep the existing nine-target angle and ten-target expression generation jobs; improve their validation, metadata, and downstream training selection.
- Add a dataset-purpose manifest/filter before changing LoRA training. Identity training should include only approved identity references by default; expression/angle/body data should be included only in the explicitly selected compatible trainer.
- Preserve the global body LoRA as the current body-type foundation. First improve its data quality, labels/captions, independent evaluation, and generation controls. Only prototype 3D body geometry after model/license/resource review.
- Integrate PromptEngine into the active request path before claiming prompt enhancement is operational.
- Connect the learned preference selector to validated prompt strategy arms, and include profile/model/task isolation in the context key.
- Make feedback observation persistence and reward-state updates concurrency-safe and idempotent.
- Keep automatic diffusion-weight RL out of scope; the first reinforcement layer learns which validated prompt/configuration strategies work best.

## Verification limitations

This audit used the connected GitHub repository's file-reading capability. It did not execute the test suite or run the application against a configured PostgreSQL database or GPU worker. Phase 0's source inspection is complete; runtime baseline tests remain pending in the developer's local environment or a configured CI runner.
