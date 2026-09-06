"""Защита входа методиста от подбора пароля и общего токена.

Регресс (2026-09-05): ни `/api/methodist/account/login`, ни путь общего
`METHODIST_TOKEN` не ограничивали неудачные попытки, а сам токен сверялся
обычным `==` - то есть за время ответа подбирался по символам.
"""

from __future__ import annotations

import pytest

from clinical_knowledge.auth_throttle import AuthThrottle, client_key


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def throttle_env(monkeypatch):
    monkeypatch.setenv("AUTH_THROTTLE_ENABLED", "1")
    monkeypatch.setenv("AUTH_THROTTLE_MAX_FAILURES", "3")
    monkeypatch.setenv("AUTH_THROTTLE_WINDOW_SEC", "100")
    monkeypatch.setenv("AUTH_THROTTLE_LOCKOUT_SEC", "60")


def test_locks_after_max_failures(throttle_env):
    clock = FakeClock()
    t = AuthThrottle(clock=clock)

    assert t.register_failure("1.2.3.4") == 0
    assert t.register_failure("1.2.3.4") == 0
    assert not t.is_locked("1.2.3.4")

    # Третья неудача достигает порога.
    locked_for = t.register_failure("1.2.3.4")
    assert locked_for == 60
    assert t.is_locked("1.2.3.4")


def test_lockout_expires(throttle_env):
    clock = FakeClock()
    t = AuthThrottle(clock=clock)
    for _ in range(3):
        t.register_failure("1.2.3.4")
    assert t.is_locked("1.2.3.4")

    clock.advance(59)
    assert t.is_locked("1.2.3.4"), "блокировка не должна истекать раньше срока"

    clock.advance(2)
    assert not t.is_locked("1.2.3.4")
    assert t.failure_count("1.2.3.4") == 0, "после блокировки счётчик обнуляется"


def test_success_resets_counter(throttle_env):
    clock = FakeClock()
    t = AuthThrottle(clock=clock)
    t.register_failure("1.2.3.4")
    t.register_failure("1.2.3.4")
    assert t.failure_count("1.2.3.4") == 2

    t.register_success("1.2.3.4")
    assert t.failure_count("1.2.3.4") == 0
    # После успеха снова нужны все три неудачи.
    assert t.register_failure("1.2.3.4") == 0
    assert t.register_failure("1.2.3.4") == 0
    assert t.register_failure("1.2.3.4") == 60


def test_old_failures_leave_the_window(throttle_env):
    clock = FakeClock()
    t = AuthThrottle(clock=clock)
    t.register_failure("1.2.3.4")
    t.register_failure("1.2.3.4")

    clock.advance(101)  # окно 100 сек - старые неудачи больше не считаются
    assert t.failure_count("1.2.3.4") == 0
    assert t.register_failure("1.2.3.4") == 0
    assert not t.is_locked("1.2.3.4")


def test_keys_are_independent(throttle_env):
    clock = FakeClock()
    t = AuthThrottle(clock=clock)
    for _ in range(3):
        t.register_failure("1.2.3.4")
    assert t.is_locked("1.2.3.4")
    assert not t.is_locked("5.6.7.8"), "блокировка одного адреса не трогает другой"


def test_disabled_throttle_never_locks(monkeypatch):
    monkeypatch.setenv("AUTH_THROTTLE_ENABLED", "0")
    monkeypatch.setenv("AUTH_THROTTLE_MAX_FAILURES", "1")
    t = AuthThrottle(clock=FakeClock())
    for _ in range(10):
        assert t.register_failure("1.2.3.4") == 0
    assert not t.is_locked("1.2.3.4")


def test_client_key_prefers_forwarded_for():
    # За Caddy реальный адрес приходит в X-Forwarded-For.
    assert client_key({"x-forwarded-for": "203.0.113.9, 10.0.0.1"}) == "203.0.113.9"
    assert client_key({"x-real-ip": "203.0.113.5"}) == "203.0.113.5"
    assert client_key({}, fallback="127.0.0.1") == "127.0.0.1"


def test_methodist_token_comparison_is_constant_time(monkeypatch):
    """verify_methodist_token не должен выходить на первом несовпадении."""
    import inspect

    from clinical_knowledge import feedback_store

    source = inspect.getsource(feedback_store.verify_methodist_token)
    assert "compare_digest" in source, (
        "сверка токена обязана идти через hmac.compare_digest, иначе токен "
        "подбирается по времени ответа"
    )

    monkeypatch.setenv("METHODIST_TOKEN", "correct-horse-battery-staple")
    assert feedback_store.verify_methodist_token("correct-horse-battery-staple")
    assert not feedback_store.verify_methodist_token("correct-horse-battery-stapl")
    assert not feedback_store.verify_methodist_token("wrong")
    assert not feedback_store.verify_methodist_token("")
    assert not feedback_store.verify_methodist_token(None)


def test_no_token_configured_rejects_everything(monkeypatch):
    from clinical_knowledge import feedback_store

    monkeypatch.delenv("METHODIST_TOKEN", raising=False)
    monkeypatch.delenv("METHODIST_PIN", raising=False)
    assert not feedback_store.verify_methodist_token("anything")
    assert not feedback_store.methodist_auth_enabled()
