# Generation feedback, rewards, and penalties

## Implemented service foundation

- Dimension-aware scoring for identity, face angle, expression, body proportions,
  technical image quality, and prompt adherence.
- Human ratings accept 1–5 or normalized -1..1 values.
- Existing pose validation contributes only to angle.
- Existing technical quality checks contribute only to technical quality.
- Missing measurements remain unscored and are marked for human review.
- Contextual UCB1 preference learning separates task/model/backend/control contexts.
- PostgreSQL persistence stores feedback observations and bandit state.

## Honest limits

MediaPipe pose estimation is approximate. It can inform angle scoring but does
not establish identity, expression, or body-proportion correctness. The current
technical quality checker measures resolution, blur, and exposure; it does not
measure anatomy or realism. Identity, expression, body proportions, and prompt
adherence therefore require explicit user ratings until dedicated evaluators
are installed and validated.

This is a contextual bandit for selecting among pre-registered, validated
generation configurations. It does not update diffusion weights, perform
gradient-based reinforcement learning, or guarantee better images. Feedback
must never override model compatibility, GPU resource limits, or safeguards.

## Persistence

Set DATABASE_URL to the existing Supabase/PostgreSQL URI. The first call to
ensure_feedback_schema() creates generation_feedback and generation_bandit_state.
Do not store raw face images or model weights in these feedback tables.

## Service API

evaluate_generation(...) creates a transparent score report.
record_generation_feedback(...) stores a report and updates its contextual
bandit state. Call it after an asset has been saved, using its saved asset ID
and the exact generation configuration ID used for that output.

## Application integration status

The Flask app now exposes POST feedback routes for saved generated assets and
face-training photos. The media library and face-dataset photo cards expose
1–5 rating controls for the six dimensions above. Feedback is persisted with
asset/profile/task/configuration context and updates the matching contextual
bandit state.

The current saved asset schema does not preserve every numeric sampler setting
(CFG, steps, scheduler, seed, ControlNet/IP-Adapter scales). Therefore the
bandit currently learns from a coarse configuration label (generation mode +
backend class), not a complete parameter vector. It does not yet change sampler
settings automatically. Before enabling automatic parameter tuning, persist
the exact generation configuration and register a bounded set of validated
candidate configurations. The service intentionally does not guess those
settings or silently alter generation behavior.
