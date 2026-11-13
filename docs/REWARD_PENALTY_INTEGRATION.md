# Reward and penalty integration

## What is connected

- Every saved image/video attempts a technical-quality evaluation and stores a reward observation linked to the asset.
- The media library lets a user submit 1–5 ratings for visual quality, identity match, and prompt adherence. Ratings are normalized to [-1, 1].
- Local LoRA training records a terminal reward/penalty observation using finite training-loss samples from the trainer log.
- Cloud LoRA training records a terminal observation when the worker reports completed or failed. If the worker does not return loss observations, the result is explicitly marked as needing review rather than receiving a fabricated score.
- The training status panel displays the reward, penalties, and dimensions still requiring human review.

## Scoring rules

- Automatic image scoring only covers the existing conservative technical image checks. It does not claim identity fidelity or prompt adherence.
- Training loss is a diagnostic only. Automatic training reward is capped at +0.20 and a completed run still requires human review of identity and sample quality.
- Failed training receives a -1.0 stability score and a non-promotion penalty.
- Invalid/non-finite training loss observations are penalized.
- Missing dimensions remain in `needs_human_review`; they are never silently treated as zero or as a pass.
- Feedback is partitioned by task, model family, backend class, and control mode in the contextual bandit.

## Safety and model integrity

The reward system only ranks observed configurations. It does **not** modify diffusion weights, LoRA weights, optimizer settings, sampler settings, prompt text, model loading, or GPU routing. It does not automatically approve or promote a trained LoRA. Human review and the existing model-version workflow remain authoritative.

## Storage and failure behavior

Feedback is stored in the existing PostgreSQL `generation_feedback` and `generation_bandit_state` tables through the existing feedback store. The feedback schema is created lazily. If feedback persistence fails, a successfully generated/saved asset or completed training run remains successful; the failure is logged and does not corrupt the generation/training result.

## Validation still required

Run the project's Python test suite and test the UI against a configured PostgreSQL database. Exercise one local generation, one video generation, one successful LoRA training run, one failed training run, and manual ratings. GPU runtime and cloud worker behavior cannot be validated from the GitHub source editor alone.
