"""Canvas composition: grids, labels, legends, headers."""
from __future__ import annotations

from PIL import Image, ImageDraw

from . import style

GAP = 6
HEADER_H = 150
FOOTER_H = 120
ROW_LABEL_W = 250
COL_LABEL_H = 56


def compose_grid(cells: dict, rows: int, cols: int, row_labels: list, col_labels: list,
                 title: str, subtitle: str, cell_size: int | None = None,
                 footer: str | None = None) -> Image.Image:
    sample = next(iter(cells.values()))
    cw, ch = sample.size
    width = ROW_LABEL_W + cols * (cw + GAP) + GAP
    height = HEADER_H + COL_LABEL_H + rows * (ch + GAP) + GAP + FOOTER_H
    canvas = Image.new("RGB", (width, height), style.BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 24), title, fill=style.INK, font=style.font(44, hand=True))
    draw.text((28, 84), subtitle, fill=style.MUTED, font=style.font(24, hand=False))
    for col, label in enumerate(col_labels):
        x = ROW_LABEL_W + col * (cw + GAP) + GAP
        draw.text((x + 8, HEADER_H + 12), label, fill=style.INK, font=style.font(34, hand=True))
    for row, label in enumerate(row_labels):
        y = HEADER_H + COL_LABEL_H + row * (ch + GAP) + GAP
        draw.text((20, y + ch // 2 - 20), label, fill=style.INK, font=style.font(30, hand=True))
        draw.line((ROW_LABEL_W - 12, y, ROW_LABEL_W - 12, y + ch), fill=(210, 214, 218), width=2)
    for (row, col), image in cells.items():
        x = ROW_LABEL_W + col * (cw + GAP) + GAP
        y = HEADER_H + COL_LABEL_H + row * (ch + GAP) + GAP
        canvas.paste(image, (x, y))
    if footer:
        draw.text((28, height - FOOTER_H + 30), footer, fill=style.MUTED, font=style.font(22, hand=False))
    return canvas


def legend_strip(width: int) -> Image.Image:
    """Footer legend: temporal gradient, contact, visibility, distance scales."""
    height = 96
    strip = Image.new("RGB", (width, height), style.BACKGROUND)
    draw = ImageDraw.Draw(strip)
    x = 28
    y = 18
    draw.text((x, y), "time", fill=style.INK, font=style.font(24, hand=True))
    import numpy as np
    stops = np.vstack([style.LIGHT[0] * 255, style.DARK[0] * 255]).astype(np.float32)
    bar = style.gradient_bar(180, 22, stops)
    strip.paste(bar, (x + 60, y + 2))
    draw.text((x + 250, y), "earlier → later", fill=style.MUTED, font=style.font(20, hand=False))
    x = 470
    for label, stops in (("contact", style.CONTACT_STOPS), ("visibility", style.VISIBILITY_STOPS),
                         ("distance 0–5 cm", style.DISTANCE_STOPS)):
        draw.text((x, y), label, fill=style.INK, font=style.font(24, hand=True))
        bar = style.gradient_bar(160, 22, stops)
        strip.paste(bar, (x + 110, y + 2))
        x += 330
    return strip
