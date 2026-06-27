#!/usr/bin/env python3
"""SVG-лого protocol в стиле protocol_logo.png: serif lowercase + золотой крест-t."""
from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF_LOGO = ROOT / "protocol_logo.png"
FONT_URL = (
    "https://fonts.gstatic.com/s/librebaskerville/v24/"
    "kmKUZrc3Hgbbcjq75U4uslyuy4kn0olVQ-LglH6T17ujFgkSCQ.ttf"
)
FONT_CACHE = ROOT / "scripts" / ".cache" / "LibreBaskerville-Bold.ttf"
TEXT_COLOR = "#09443D"
GOLD_TOP = "#F2D078"
GOLD_MID = "#D4AF37"
GOLD_BOTTOM = "#A67C00"
FONT_SIZE = 96
BASELINE_Y = 96.0
CROSS_OX, CROSS_OY = 918, 221
REF_CROSS_H = 245.0


@dataclass
class PathPart:
    d: str


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit("pip install opencv-python-headless") from exc
    return cv2


def _require_fonttools():
    try:
        from fontTools.ttLib import TTFont  # type: ignore
        from fontTools.misc.transform import Transform  # type: ignore
        from fontTools.pens.svgPathPen import SVGPathPen  # type: ignore
        from fontTools.pens.transformPen import TransformPen  # type: ignore
    except ImportError as exc:
        raise SystemExit("pip install fonttools") from exc
    return TTFont, Transform, SVGPathPen, TransformPen


def _contour_path(c, ox: int, oy: int, simplify: float = 1.0) -> str:
    import cv2
    import numpy as np

    approx = cv2.approxPolyDP(c, simplify, True)
    pts = approx.reshape(-1, 2) - np.array([ox, oy])
    d = f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"
    for x, y in pts[1:]:
        d += f" L {x:.2f} {y:.2f}"
    return d + " Z"


def extract_gold_cross_paths(ref_path: Path) -> tuple[list[PathPart], int, int]:
    cv2 = _require_cv2()
    import numpy as np

    img = cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(ref_path)
    if img.ndim == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        b, g, r = cv2.split(img)
    else:
        b, g, r, _a = cv2.split(img)

    gold = ((r > 130) & (g > 90) & (b < 130) & (r > g - 20)).astype(np.uint8) * 255
    roi = gold.copy()
    roi[:, :850] = 0
    roi[:, 1120:] = 0
    roi[:180, :] = 0
    roi[480:, :] = 0
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    parts: list[PathPart] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 800:
            continue
        parts.append(PathPart(_contour_path(c, CROSS_OX, CROSS_OY, 1.0)))
        if len(parts) >= 2:
            break
    if len(parts) < 2:
        raise SystemExit("could not trace gold cross from protocol_logo.png")

    ys, xs = np.where(roi)
    w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    return parts, w, h


def _ensure_font() -> Path:
    if not FONT_CACHE.is_file():
        FONT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FONT_URL, FONT_CACHE)
    return FONT_CACHE


def _text_paths(text: str, x: float, baseline: float, font_size: float) -> tuple[list[str], float]:
    TTFont, Transform, SVGPathPen, TransformPen = _require_fonttools()
    font = TTFont(_ensure_font())
    gs = font.getGlyphSet()
    scale = font_size / font["head"].unitsPerEm
    paths: list[str] = []
    cx = x
    cmap = font.getBestCmap()
    for ch in text:
        gname = cmap[ord(ch)]
        pen = SVGPathPen(None)
        tpen = TransformPen(pen, Transform(scale, 0, 0, -scale, cx, baseline))
        gs[gname].draw(tpen)
        d = pen.getCommands()
        if d:
            paths.append(d)
        cx += gs[gname].width * scale
    return paths, cx


def _layout(cross_w: int, cross_h: int) -> dict[str, float]:
    pro_paths, pro_end = _text_paths("pro", 0, BASELINE_Y, FONT_SIZE)
    _, total_w = _text_paths("protocol", 0, BASELINE_Y, FONT_SIZE)
    ocol_paths, ocol_w = _text_paths("ocol", 0, BASELINE_Y, FONT_SIZE)
    t_advance = _text_paths("t", 0, BASELINE_Y, FONT_SIZE)[1]

    symbol_scale = (REF_CROSS_H / cross_h) * (FONT_SIZE / 72.0) * 0.72
    symbol_w = cross_w * symbol_scale
    symbol_h = cross_h * symbol_scale

    symbol_x = pro_end + (t_advance - symbol_w) / 2.0
    ocol_x = pro_end + t_advance + max(4.0, (symbol_w - t_advance) * 0.35)
    if ocol_x < symbol_x + symbol_w + 2:
        ocol_x = symbol_x + symbol_w + 4

    symbol_y = BASELINE_Y - symbol_h * 0.78
    view_top = symbol_y - 8
    total_h = BASELINE_Y - view_top + 12
    total_width = max(ocol_x + ocol_w + 8, total_w + 8)

    return {
        "pro_paths": pro_paths,
        "ocol_paths": ocol_paths,
        "pro_end": pro_end,
        "ocol_x": ocol_x,
        "ocol_w": ocol_w,
        "symbol_scale": symbol_scale,
        "symbol_x": symbol_x,
        "symbol_y": symbol_y,
        "symbol_h": symbol_h,
        "view_top": view_top,
        "total_h": total_h,
        "total_w": total_width,
    }


def _svg_document(
    cross_parts: list[PathPart],
    cross_h: int,
    layout: dict[str, float],
) -> str:
    cross_markup = "\n      ".join(f'<path d="{p.d}"/>' for p in cross_parts)
    pro_markup = "\n".join(f'  <path d="{d}" fill="{TEXT_COLOR}"/>' for d in layout["pro_paths"])
    ocol_markup = "\n".join(
        f'  <path d="{d}" fill="{TEXT_COLOR}" transform="translate({layout["ocol_x"]:.2f}, 0)"/>'
        for d in layout["ocol_paths"]
    )
    grad_h = layout["symbol_h"]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 {layout["view_top"]:.1f} {layout["total_w"]:.1f} {layout["total_h"]:.1f}" role="img" aria-label="protocol">
  <defs>
    <linearGradient id="protocol-gold" x1="0" y1="0" x2="0" y2="{grad_h:.1f}" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="{GOLD_TOP}"/>
      <stop offset="42%" stop-color="{GOLD_MID}"/>
      <stop offset="100%" stop-color="{GOLD_BOTTOM}"/>
    </linearGradient>
    <g id="protocol-gold-t">
      {cross_markup}
    </g>
  </defs>
{pro_markup}
  <g transform="translate({layout["symbol_x"]:.2f} {layout["symbol_y"]:.2f}) scale({layout["symbol_scale"]:.6f})" fill="url(#protocol-gold)">
    <use href="#protocol-gold-t"/>
  </g>
{ocol_markup}
</svg>
"""


def write_svgs(cross_parts: list[PathPart], cross_w: int, cross_h: int) -> None:
    layout = _layout(cross_w, cross_h)
    doc = _svg_document(cross_parts, cross_h, layout)
    for name in ("protocol_logo.svg", "protocol-wordmark.svg"):
        (ROOT / name).write_text(doc, encoding="utf-8")

    cross_only = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {cross_w} {cross_h}" role="img" aria-label="protocol t">
  <defs>
    <linearGradient id="protocol-gold" x1="0" y1="0" x2="0" y2="{cross_h}" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="{GOLD_TOP}"/>
      <stop offset="42%" stop-color="{GOLD_MID}"/>
      <stop offset="100%" stop-color="{GOLD_BOTTOM}"/>
    </linearGradient>
  </defs>
  <g fill="url(#protocol-gold)">
    {" ".join(f'<path d="{p.d}"/>' for p in cross_parts)}
  </g>
</svg>
"""
    (ROOT / "protocol-t-symbol.svg").write_text(cross_only, encoding="utf-8")


def write_png_previews() -> None:
    try:
        import cairosvg  # type: ignore
    except ImportError:
        print("skip PNG: pip install cairosvg", file=sys.stderr)
        return
    for svg_name, png_name, width in (
        ("protocol_logo.svg", "protocol_logo_render.png", 1600),
        ("protocol-wordmark.svg", "protocol-wordmark.png", 1600),
        ("protocol-t-symbol.svg", "protocol-t-symbol.png", 360),
    ):
        cairosvg.svg2png(url=str(ROOT / svg_name), write_to=str(ROOT / png_name), output_width=width)


def main() -> None:
    if not REF_LOGO.is_file():
        raise SystemExit(f"missing reference: {REF_LOGO}")
    parts, cw, ch = extract_gold_cross_paths(REF_LOGO)
    write_svgs(parts, cw, ch)
    write_png_previews()
    print(f"ok: protocol_logo.svg ({len(parts)} cross paths, serif lowercase, gold gradient)")


if __name__ == "__main__":
    main()
