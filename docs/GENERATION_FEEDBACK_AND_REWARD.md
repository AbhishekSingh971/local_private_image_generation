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

## Remaining application-shell wiring

The Flask application and browser UI are not yet wired to call these services.
An authenticated/CSRF-protected feedback route, UI controls on generated assets,
and generation-configuration metadata capture are required before live feedback
can influence future image or face-dataset generation. Do not infer a
configuration ID from the prompt or silently select unvalidated settings.
