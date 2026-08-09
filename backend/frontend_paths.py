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
    "mo-calibration.html": "methodist/mo-calibration.html",
    "expert.html": "methodist/expert.html",
    "onco-risk.html": "methodist/onco-risk.html",
    # Patient workspace
    "patient.html": "patient/patient.html",
    "patient-check.html": "patient/patient-check.html",
    "patient-tokens.css": "patient/patient-tokens.css",
    "patient-ui.js": "patient/patient-ui.js",
    "patient-manifest.webmanifest": "patient/patient-manifest.webmanifest",
    "patient-sw.js": "patient/patient-sw.js",
    # Shared tools / doctor chrome / brand / МО assets
    "proto-viewer.html": "shared/proto-viewer.html",
    "protocol-chrome-tabs.css": "shared/protocol-chrome-tabs.css",
    "search-flow.css": "shared/search-flow.css",
    "search-flow.js": "shared/search-flow.js",
    "ux-redesign.css": "shared/ux-redesign.css",
    "protocol-logo.svg": "shared/protocol-logo.svg",
    "protocol-logo-mini.svg": "shared/protocol-logo-mini.svg",
    "protocol-logo-wordmark.svg": "shared/protocol-logo-wordmark.svg",
    "protocol-logo-wordmark-text.svg": "shared/protocol-logo-wordmark-text.svg",
    "protocol_logo_curves_transparent.svg": "shared/protocol_logo_curves_transparent.svg",
    "logo_mini.png": "shared/logo_mini.png",
    "mo-tokens.css": "shared/mo-tokens.css",
    "methodist-cabinet.css": "shared/methodist-cabinet.css",
    "mo-protocol-viewer.css": "shared/mo-protocol-viewer.css",
    "mo-ui.css": "shared/mo-ui.css",
    "mo-api.js": "shared/mo-api.js",
    "mo-charts.js": "shared/mo-charts.js",
    "mo-app.js": "shared/mo-app.js",
    "mo-calibration.css": "shared/mo-calibration.css",
    "mo-calibration.js": "shared/mo-calibration.js",
    "vendor/echarts.min.js": "shared/vendor/echarts.min.js",
    "vendor/ECHARTS-LICENSE.txt": "shared/vendor/ECHARTS-LICENSE.txt",
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

