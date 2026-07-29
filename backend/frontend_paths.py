"""Frontend asset path resolution with backward compatibility.

Canonical target layout:
    frontend/web/<asset>

Compatibility:
    if canonical file is missing, fallback to repository root (<asset>).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_WEB_ROOT = ROOT / "frontend" / "web"


def frontend_file(name: str) -> Path:
    """Resolve frontend file path: canonical first, legacy root second."""
    canonical = FRONTEND_WEB_ROOT / name
    if canonical.is_file():
        return canonical
    return ROOT / name


def has_frontend_file(name: str) -> bool:
    return frontend_file(name).is_file()

