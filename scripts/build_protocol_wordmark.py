#!/usr/bin/env python3
"""Собрать wordmark Protocol: Pro + T(крест с обводкой + лепестки) + ocol."""
from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FONT_URL = (
    "https://fonts.gstatic.com/s/dmsans/v17/"
    "rP2tp2ywxg089UriI5-g4vlH9VoD8CmcqZG40F9JadbnoEwARZthTg.ttf"
)
FONT_CACHE = ROOT / "scripts" / ".cache" / "DMSans-Bold.ttf"
SYMBOL_GREEN = "#00AB68"
TEXT_COLOR = "#063d35"
OUTLINE_WHITE = "#FFFFFF"
FONT_SIZE = 72
PETAL_SPLIT_Y = 115
# из logo_mini: перекladина креста и низ ножки T (нормализовано по высоте символа)
CROSSBAR_Y_RATIO = 167 / 239
STEM_BOTTOM_Y_RATIO = 238 / 239
T_CROSSBAR_Y = 26.0
BASELINE_Y = 72.0


@dataclass
class SymbolPart:
    d: str
    fill: str


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


def _contour_path(c, ox: int, oy: int, simplify: float = 0.35) -> str:
    import cv2
    import numpy as np

    approx = cv2.approxPolyDP(c, simplify, True)
    pts = approx.reshape(-1, 2) - np.array([ox, oy])
    d = f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"
    for x, y in pts[1:]:
        d += f" L {x:.2f} {y:.2f}"
    return d + " Z"


def extract_t_symbol(logo_path: Path) -> tuple[list[SymbolPart], int, int]:
    """Белая обводка креста, зелёный крест (буква T), три лепестка сверху."""
    cv2 = _require_cv2()
    import numpy as np

    img = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(logo_path)
    b, g, r, a = cv2.split(img)
    green = (
        (g >= 168) & (g <= 175) & (r <= 5) & (b >= 100) & (b <= 108) & (a > 200)
    ).astype(np.uint8) * 255

    ys, xs = np.where(green)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    petal_mask = np.zeros_like(green)
    petal_mask[:PETAL_SPLIT_Y, :] = green[:PETAL_SPLIT_Y, :]
    cross_green = np.zeros_like(green)
    cross_green[PETAL_SPLIT_Y:, :] = green[PETAL_SPLIT_Y:, :]

    kernel = np.ones((5, 5), np.uint8)
    cross_outer = cv2.dilate(cross_green, kernel, iterations=1)

    parts: list[SymbolPart] = []

    cnts, _ = cv2.findContours(cross_outer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        parts.append(SymbolPart(_contour_path(max(cnts, key=cv2.contourArea), x0, y0), OUTLINE_WHITE))

    cnts, _ = cv2.findContours(cross_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        parts.append(SymbolPart(_contour_path(max(cnts, key=cv2.contourArea), x0, y0), SYMBOL_GREEN))

    cnts, _ = cv2.findContours(petal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
        if cv2.contourArea(c) < 200:
            continue
        parts.append(SymbolPart(_contour_path(c, x0, y0), SYMBOL_GREEN))

    sw, sh = x1 - x0 + 1, y1 - y0 + 1
    return parts, sw, sh


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
    for ch in text:
        gname = font.getBestCmap()[ord(ch)]
        pen = SVGPathPen(None)
        tpen = TransformPen(pen, Transform(scale, 0, 0, -scale, cx, baseline))
        gs[gname].draw(tpen)
        d = pen.getCommands()
        if d:
            paths.append(d)
        cx += gs[gname].width * scale
    return paths, cx


def _layout_symbol(sw: int, sh: int) -> tuple[float, float, float, float]:
    """symbol_h, symbol_x, symbol_y, view_top."""
    stem_span = STEM_BOTTOM_Y_RATIO - CROSSBAR_Y_RATIO
    symbol_h = (BASELINE_Y - T_CROSSBAR_Y) / stem_span
    symbol_w = symbol_h * (sw / sh)
    symbol_y = T_CROSSBAR_Y - CROSSBAR_Y_RATIO * symbol_h
    view_top = symbol_y - 4
    return symbol_h, symbol_w, symbol_y, view_top


def _part_markup(parts: list[SymbolPart]) -> str:
    return "\n".join(f'      <path d="{p.d}" fill="{p.fill}"/>' for p in parts)


def write_svgs(parts: list[SymbolPart], sw: int, sh: int) -> None:
    markup = _part_markup(parts)
    symbol_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {sw} {sh}" role="img" aria-label="Protocol T">
  <g>
{markup}
  </g>
</svg>
"""
    (ROOT / "protocol-t-symbol.svg").write_text(symbol_svg, encoding="utf-8")

    pro_paths, pro_end = _text_paths("Pro", 0, BASELINE_Y, FONT_SIZE)
    ocol_paths, ocol_w = _text_paths("ocol", 0, BASELINE_Y, FONT_SIZE)

    symbol_h, symbol_w, symbol_y, view_top = _layout_symbol(sw, sh)
    gap = 4
    symbol_x = pro_end + gap
    ocol_x = symbol_x + symbol_w + gap
    total_w = ocol_x + ocol_w + 4
    total_h = BASELINE_Y - view_top + 8

    pro_markup = "\n".join(f'  <path d="{d}" fill="{TEXT_COLOR}"/>' for d in pro_paths)
    ocol_markup = "\n".join(
        f'  <path d="{d}" fill="{TEXT_COLOR}" transform="translate({ocol_x:.2f}, 0)"/>'
        for d in ocol_paths
    )

    wordmark = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 {view_top:.1f} {total_w:.1f} {total_h:.1f}" role="img" aria-label="Protocol">
  <defs>
    <g id="protocol-t-mark">
{markup}
    </g>
  </defs>
{pro_markup}
  <g transform="translate({symbol_x:.2f} {symbol_y:.2f}) scale({symbol_h / sh:.6f})">
    <use href="#protocol-t-mark"/>
  </g>
{ocol_markup}
</svg>
"""
    (ROOT / "protocol-wordmark.svg").write_text(wordmark, encoding="utf-8")


def write_png_previews() -> None:
    try:
        import cairosvg  # type: ignore
    except ImportError:
        print("skip PNG: pip install cairosvg", file=sys.stderr)
        return
    for name in ("protocol-wordmark.svg", "protocol-t-symbol.svg"):
        svg = ROOT / name
        png = ROOT / name.replace(".svg", ".png")
        cairosvg.svg2png(
            url=str(svg),
            write_to=str(png),
            output_width=1200 if "wordmark" in name else 400,
        )


def main() -> None:
    logo = ROOT / "logo_mini.png"
    parts, sw, sh = extract_t_symbol(logo)
    if len(parts) < 5:
        raise SystemExit(f"expected 5 symbol parts (outline+cross+3 petals), got {len(parts)}")
    write_svgs(parts, sw, sh)
    write_png_previews()
    print(f"ok: {len(parts)} parts, symbol {sw}x{sh}, DM Sans Bold paths, cross with white outline")


if __name__ == "__main__":
    main()
