"""Защита входа от подбора пароля и токена.

Зачем: до 2026-09-05 ни `/api/methodist/account/login`, ни путь общего
`METHODIST_TOKEN` не ограничивали число неудачных попыток. Общий rate-limit в
`rag_server.py` давал 60 запросов в минуту на маршрут, то есть подбор короткого
токена или пароля был вопросом времени и ничем не выделялся в логах.

Модель: скользящее окно неудач по ключу (обычно IP клиента). Когда неудач
становится больше порога, ключ блокируется на фиксированное время; успешный
вход счётчик обнуляет.

Состояние в памяти процесса. Для одного контейнера на GCE этого достаточно;
при переходе на несколько реплик счётчик нужно переносить в общий Redis - тогда
лимит станет общим, а не «на каждый процесс свой».

Настройки:
    AUTH_THROTTLE_ENABLED       - включён (по умолчанию да)
    AUTH_THROTTLE_MAX_FAILURES  - неудач до блокировки (по умолчанию 10)
    AUTH_THROTTLE_WINDOW_SEC    - окно наблюдения, сек (по умолчанию 900)
    AUTH_THROTTLE_LOCKOUT_SEC   - длительность блокировки, сек (по умолчанию 900)
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def max_failures() -> int:
    return _env_int("AUTH_THROTTLE_MAX_FAILURES", 10)


def window_sec() -> float:
    return float(_env_int("AUTH_THROTTLE_WINDOW_SEC", 900))


def lockout_sec() -> float:
    return float(_env_int("AUTH_THROTTLE_LOCKOUT_SEC", 900))


def enabled() -> bool:
    return _env_bool("AUTH_THROTTLE_ENABLED", True)


@dataclass
class _Entry:
    failures: list[float] = field(default_factory=list)
    locked_until: float = 0.0


class AuthThrottle:
    """Счётчик неудачных попыток входа с блокировкой по ключу."""

    def __init__(self, *, clock=time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}

    def retry_after(self, key: str) -> int:
        """Сколько секунд ключ ещё заблокирован. 0 - вход разрешён."""
        if not enabled():
            return 0
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return 0
            if entry.locked_until > now:
                return max(1, int(entry.locked_until - now))
            if entry.locked_until:
                # Блокировка истекла - начинаем с чистого листа.
                self._entries.pop(key, None)
            return 0

    def is_locked(self, key: str) -> bool:
        return self.retry_after(key) > 0

    def register_failure(self, key: str) -> int:
        """Учесть неудачу. Возвращает retry_after, если ключ заблокирован."""
        if not enabled():
            return 0
        now = self._clock()
        horizon = now - window_sec()
        limit = max_failures()
        with self._lock:
            entry = self._entries.setdefault(key, _Entry())
            if entry.locked_until > now:
                return max(1, int(entry.locked_until - now))
            entry.failures = [t for t in entry.failures if t > horizon]
            entry.failures.append(now)
            if len(entry.failures) >= limit:
                entry.locked_until = now + lockout_sec()
                entry.failures.clear()
                return max(1, int(lockout_sec()))
            return 0

    def register_success(self, key: str) -> None:
        if not enabled():
            return
        with self._lock:
            self._entries.pop(key, None)

    def failure_count(self, key: str) -> int:
        now = self._clock()
        horizon = now - window_sec()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return 0
            return len([t for t in entry.failures if t > horizon])

    def reset(self) -> None:
        """Только для тестов и обслуживания."""
        with self._lock:
            self._entries.clear()


# Отдельные счётчики: подбор пароля и подбор общего токена - разные поверхности,
# и блокировка одной не должна закрывать другую.
login_throttle = AuthThrottle()
token_throttle = AuthThrottle()


def client_key(headers, fallback: str = "unknown") -> str:
    """Ключ клиента для счётчика.

    Приложение стоит за Caddy на том же хосте, поэтому реальный адрес приходит
    в X-Forwarded-For. Берём левый элемент цепочки.
    """
    try:
        xff = (headers.get("x-forwarded-for") or "").strip()
    except AttributeError:
        return fallback
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    real = (headers.get("x-real-ip") or "").strip()
    return real or fallback
