import os
import uuid

import pytest

from prompt_intelligence.feedback_store import (
    load_bandit_state,
    save_feedback_and_bandit_state,
)
from prompt_intelligence.reward import PreferenceBandit


pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="PostgreSQL integration test requires DATABASE_URL")


def test_feedback_and_bandit_state_commit_together():
    suffix = uuid.uuid4().hex
    asset_id = suffix
    profile_id = uuid.uuid4().hex
    context = {
        "task": "image",
        "model_family": "sd15",
        "backend_class": "local_gpu",
        "control_mode": "txt2img",
        "profile_id": profile_id,
        "strategy_version": "v2",
    }
    context_key = PreferenceBandit.context_key(context)
    report = {
        "reward": 0.75,
        "scores": {"identity": 1.0},
        "sources": {"identity": "human"},
        "penalties": [],
        "needs_human_review": [],
    }
    result = save_feedback_and_bandit_state(
        asset_id=asset_id,
        profile_id=profile_id,
        source_id=asset_id,
        source_type="asset",
        task="image",
        configuration_id="identity_guard",
        context=context,
        report=report,
        context_key=context_key,
    )
    assert result["feedback_id"]
    state = load_bandit_state(context_key)
    assert state is not None
    arm = state["contexts"][context_key]["identity_guard"]
    assert arm["trials"] == 1
    assert arm["mean_reward"] == pytest.approx(0.75)
