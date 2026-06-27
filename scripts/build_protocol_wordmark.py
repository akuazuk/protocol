#!/usr/bin/env python3
"""Собрать wordmark Protocol: буква t = крест + лепестки из logo_mini.png."""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYMBOL_COLOR = "#00AB68"
TEXT_COLOR = "#063d35"
FONT_SIZE = 72


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit("pip install opencv-python-headless pillow") from exc
    return cv2


def _require_pil():
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise SystemExit("pip install pillow") from exc
    return Image


def extract_symbol_paths(logo_path: Path) -> tuple[list[str], int, int]:
    cv2 = _require_cv2()
    import numpy as np

    img = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(logo_path)
    b, g, r, a = cv2.split(img)
    mask = ((g >= 168) & (g <= 175) & (r <= 5) & (b >= 100) & (b <= 108) & (a > 200)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
    ys, xs = np.where(mask > 0)
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()), int(ys.max())
    sw, sh = x1 - x0 + 1, y1 - y0 + 1
    paths: list[str] = []
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.35, True)
        pts = approx.reshape(-1, 2) - np.array([x0, y0])
        d = f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"
        for x, y in pts[1:]:
            d += f" L {x:.2f} {y:.2f}"
        d += " Z"
        paths.append(d)
    return paths, sw, sh


def write_svgs(paths: list[str], sw: int, sh: int) -> None:
    paths_markup = "\n      ".join(f'<path d="{d}"/>' for d in paths)
    symbol_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {sw} {sh}" role="img" aria-label="Protocol mark">
  <g fill="{SYMBOL_COLOR}">
    {paths_markup}
  </g>
</svg>
"""
    (ROOT / "protocol-t-symbol.svg").write_text(symbol_svg, encoding="utf-8")

    pro_w = 98
    symbol_scale_h = 58
    symbol_w = symbol_scale_h * (sw / sh)
    symbol_x = pro_w + 1
    ocol_x = symbol_x + symbol_w - 2
    total_w = ocol_x + 168
    total_h = 78
    text_y = 58
    symbol_y = total_h - symbol_scale_h - 4
    wordmark = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w:.0f} {total_h:.0f}" role="img" aria-label="Protocol">
  <defs>
    <g id="protocol-t-mark" fill="{SYMBOL_COLOR}">
      {paths_markup}
    </g>
  </defs>
  <text x="0" y="{text_y}" font-family="Outfit, system-ui, sans-serif" font-weight="800" font-size="{FONT_SIZE}" fill="{TEXT_COLOR}" letter-spacing="-0.03em">Pro</text>
  <g transform="translate({symbol_x:.1f} {symbol_y:.1f}) scale({symbol_scale_h / sh:.6f})">
    <use href="#protocol-t-mark"/>
  </g>
  <text x="{ocol_x:.1f}" y="{text_y}" font-family="Outfit, system-ui, sans-serif" font-weight="800" font-size="{FONT_SIZE}" fill="{TEXT_COLOR}" letter-spacing="-0.03em">ocol</text>
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
    paths, sw, sh = extract_symbol_paths(logo)
    if len(paths) != 4:
        raise SystemExit(f"expected 4 symbol paths, got {len(paths)}")
    write_svgs(paths, sw, sh)
    write_png_previews()
    print(f"ok: {len(paths)} paths from {logo.name} ({sw}x{sh})")


if __name__ == "__main__":
    main()
