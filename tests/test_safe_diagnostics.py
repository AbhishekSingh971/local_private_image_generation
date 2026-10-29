import pytest

from safe_diagnostics import format_exception_detail


def test_redacts_postgres_connection_credentials():
    detail = format_exception_detail(
        RuntimeError("could not connect to postgresql://studio:supersecret@db.example.test:5432/app")
    )
    assert "supersecret" not in detail
    assert "studio:supersecret" not in detail
    assert "postgresql://[redacted]@db.example.test:5432/app" in detail


def test_redacts_named_secrets_and_tokens():
    detail = format_exception_detail(
        RuntimeError("password=hunter2, token:abc123 api_key=sk-live-value")
    )
    for secret in ("hunter2", "abc123", "sk-live-value"):
        assert secret not in detail
    assert detail.count("[redacted]") == 3


def test_bounds_and_flattens_multiline_diagnostics():
    detail = format_exception_detail(RuntimeError("line one\nline two\tline three"), limit=20)
    assert len(detail) <= 20
    assert "\n" not in detail
    assert "\t" not in detail


def test_rejects_nonpositive_limit():
    with pytest.raises(ValueError, match="limit must be positive"):
        format_exception_detail(RuntimeError("failure"), limit=0)
