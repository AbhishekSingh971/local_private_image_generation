# Unified Prompt Intelligence, Preference Learning, and Training Strategy

## Product contract

Every user-facing generation/training surface accepts one natural-language prompt.
The user is not asked to author positive or negative prompts. The backend derives
both, plus structured parameters, and passes them to the task adapter.

Surfaces covered:
- Image generation (text-to-image and image-to-image).
- Video generation.
- Face-profile training.
- Dataset-generation subsections such as face angles and expressions.

The user may still control explicit product settings such as selected face profile,
training method, output duration, compute backend, or seed. These are not duplicate
prompt fields.

## System architecture

```text
Flask UI
  |
  | prompt + task + explicit UI settings + task context
  v
Prompt API / Job Orchestrator
  |
  v
Prompt Intelligence
  |-- local/remote OpenAI-compatible structured-output provider
  |-- schema validation and intent preservation
  |-- positive prompt + negative prompt + parameters
  v
Task Adapter
  |-- image adapter
  |-- video adapter
  |-- face-training adapter
  |-- dataset-generation adapter (angle/expression context)
  v
Training / Inference Strategy
  |-- inference: base checkpoint + profile adapter + ControlNet/IP-Adapter
  |-- training: user-selected LoRA OR full fine-tuning
  v
Compute Router (CPU / local CUDA / GHive worker)
  |
  v
Quality Evaluator + User Feedback
  |-- prompt adherence
  |-- face identity consistency (where applicable)
  |-- image/video technical quality
  |-- human preference
  v
Preference Optimizer (contextual bandit initially)
  |
  +----> future parameter recommendations
  |
  v
Versioned database records and media library
```

## Prompt data contract

A prompt request contains:
- `task`: image, video, face_training, or dataset_generation.
- `prompt`: the single user-authored prompt.
- `context`: selected profile, target angle/expression, source image metadata,
  video duration/FPS, and explicit UI constraints when relevant.

The prompt engine returns:
- `positive_prompt`
- `negative_prompt`
- `parameters`: validated structured values (composition, lighting, camera,
  motion, duration, FPS, identity/pose constraints, etc.).
- `task_instructions`

The engine uses a configurable OpenAI-compatible endpoint. This can point to a
local model server or a remote provider. Configure:
- `PROMPT_LLM_BASE_URL`: complete chat-completions endpoint.
- `PROMPT_LLM_MODEL`: model identifier.
- `PROMPT_LLM_API_KEY`: optional key; keep it in .env, never commit it.

If no provider is configured, the fallback preserves the user's prompt and returns
a generic negative prompt. It does not pretend to perform semantic enhancement.
Provider failures are surfaced rather than silently generating with a degraded prompt.

## Training method selection

The training UI exposes exactly two strategies:

### LoRA
- Uses the existing `train_face_lora.py` path.
- Trains small profile-specific adapters while freezing the base model.
- Preferred for individual face identity profiles, quick iteration, and storage efficiency.
- Each profile adapter is versioned independently.

### Full fine-tuning
- A separate training strategy, not a renamed LoRA run.
- Updates selected base-model weights (UNet and optionally text encoder).
- Requires a dedicated trainer, larger VRAM/storage budget, checkpointing,
  validation split, and rollback/versioning.
- Better suited to broad domain/style adaptation with a sufficiently large,
  diverse, licensed dataset than to one person's face profile.
- The strategy selector and plan contract exist in this branch, but the full
  fine-tuning trainer is deliberately not marked executable until implemented
  and validated. The app must never silently route this choice to LoRA.

For best overall design, retain a general base/domain checkpoint and use a separate
LoRA per face identity. Full fine-tuning should be an optional model-level training
workflow, not the default for each identity.

## Preference learning and reinforcement

Start with a contextual-bandit / online preference-learning phase, not end-to-end
reinforcement learning of diffusion weights.

1. Generate a small set of candidates using prevalidated parameter configurations.
2. Evaluate technical quality and prompt adherence.
3. Ask for a simple user preference signal (keep / reject / rating).
4. Normalize feedback dimensions and compute a reward.
5. Update per-configuration reward statistics.
6. Prefer configurations with stronger historical rewards for similar task contexts.

Potential reward dimensions:
- prompt adherence,
- identity similarity,
- face visibility and pose accuracy,
- image/video technical quality,
- user preference.

Do not optimize only for an aesthetic score: it can reward attractive but incorrect
images. Keep user preference and prompt adherence separate. Do not use generated
training examples as held-out validation data.

The initial bandit module is in-memory only. Before production use, persist its
observations and configuration IDs in PostgreSQL, partitioned by task/model/profile,
and add safeguards against feedback leakage between unrelated identities.

## Model enhancement and evaluation

Model enhancement is a controlled, versioned workflow:
- validate and deduplicate datasets;
- generate/check captions and metadata;
- split train/validation sets;
- train a new checkpoint or adapter;
- evaluate on held-out prompts and images;
- compare against the previous version;
- promote only when evaluation and user review pass;
- retain rollback to the previous version.

The existing technical image checks are useful but are not an identity or prompt-
adherence metric. Add independent face-embedding similarity, pose checks, text-image
alignment, video temporal consistency, and human review where applicable. Automated
scores are signals, not proof of correctness.

## Integration sequence

1. Add prompt-engine configuration and unit tests.
2. Add task adapters and replace duplicated prompt fields in image/video/training UI.
3. Pass the structured prompt result through local and GHive job payloads.
4. Add LoRA/full-finetune selector and capability-aware validation.
5. Implement and test the separate full-finetuning trainer.
6. Add quality evaluation and persisted user feedback.
7. Add contextual-bandit recommendations and A/B evaluation.
8. Add checkpoint registry, promotion, and rollback.

## Safety and compatibility constraints

- Do not modify or delete the current LoRA trainer while adding the full-finetuning path.
- Do not change the existing generation pipeline until the new adapters pass regression tests.
- Keep SD 1.5 components and compatible ControlNet/IP-Adapter assets paired; model families
  are not interchangeable.
- Do not launch training on CPU-only hardware when the current trainer requires CUDA.
- Preserve the existing CPU inference fallback and GHive worker interface.
- Store prompts, model/checkpoint hashes, LoRA hashes, seed, scheduler, CFG, steps,
  resolution, control strengths, backend, and software versions with each job.
