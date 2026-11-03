from video_quality import summarize_temporal_metrics


def test_temporal_summary_flags_flicker_and_never_claims_identity():
    result = summarize_temporal_metrics(
        frame_count=4,
        sharpness=[30, 32, 29, 31],
        brightness=[120, 122, 119, 121],
        luminance_deltas=[2, 3, 80],
        motion_medians=[1, 1.2, 1.1],
    )
    assert result["status"] == "needs_review"
    assert result["identity_consistency"] == "not_assessed"
    assert result["human_review_required"] is True


def test_temporal_summary_accepts_stable_technical_samples_but_requires_review():
    result = summarize_temporal_metrics(
        frame_count=4,
        sharpness=[30, 32, 29, 31],
        brightness=[120, 122, 119, 121],
        luminance_deltas=[2, 3, 2],
        motion_medians=[1, 1.2, 1.1],
    )
    assert result["status"] == "technical_checks_passed"
    assert result["identity_consistency"] == "not_assessed"
    assert result["human_review_required"] is True
