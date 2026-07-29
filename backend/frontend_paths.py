"""Frontend asset path resolution with backward compatibility.

Canonical target layout:
    frontend/web/<domain>/<asset>

Compatibility:
    if canonical file is missing, fallback to repository root (<asset>).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_WEB_ROOT = ROOT / "frontend" / "web"

# Legacy root filename -> canonical grouped path under frontend/web.
CANONICAL_ASSET_MAP: dict[str, str] = {
    # Doctor / core workspace
    "index.html": "doctor/index.html",
    "consult_review.html": "doctor/consult_review.html",
    # Methodist workspace
    "mis-kz-quality.html": "methodist/mis-kz-quality.html",
    "onco-risk.html": "methodist/onco-risk.html",
    # Patient workspace
    "patient.html": "patient/patient.html",
    "patient-check.html": "patient/patient-check.html",
    "patient-tokens.css": "patient/patient-tokens.css",
    "patient-ui.js": "patient/patient-ui.js",
    "patient-manifest.webmanifest": "patient/patient-manifest.webmanifest",
    "patient-sw.js": "patient/patient-sw.js",
    # Shared tools
    "proto-viewer.html": "shared/proto-viewer.html",
}


def frontend_file(name: str) -> Path:
    """Resolve frontend file path: canonical first, legacy root second."""
    mapped = CANONICAL_ASSET_MAP.get(name, name)
    canonical = FRONTEND_WEB_ROOT / mapped
    if canonical.is_file():
        return canonical
    return ROOT / name


def has_frontend_file(name: str) -> bool:
    return frontend_file(name).is_file()

