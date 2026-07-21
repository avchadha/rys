"""Style x Content grid for layer-anatomy analysis.

Vision analog of the Sapir-Whorf experiment from the RYS blog series
(https://dnhkng.github.io/posts/sapir-whorf/): there, 8 languages x 8 topics
revealed that LLM middle layers organize by content (topic) while early/late
layers organize by format (language). Here, "language" becomes rendering
STYLE and "topic" becomes shape CONTENT.

The grid is 8 shapes x 8 styles = 64 images, one per cell, giving 2016
pairwise comparisons in three categories:
  - same content, different style   (224 pairs)  — "same topic, diff language"
  - same style, different content   (224 pairs)  — "same language, diff topic"
  - different content and style     (1568 pairs)

If middle ViT layers organize by content (shape identity) rather than style,
the same-content curve should dominate mid-stack after per-layer centering —
identifying the "reasoning" phase where RYS duplication should be tolerated.

Each (content, style) cell gets a deterministic rotation/position jitter so
same-content pairs cannot match at the pixel level; the model must abstract
shape identity across renderings.
"""

import math

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


CANVAS = 448

SHAPE_NAMES = [
    "triangle", "square", "pentagon", "hexagon",
    "star", "circle", "cross", "arrow",
]

STYLE_NAMES = [
    "solid_white_bg", "outline", "solid_black_bg", "stipple_fill",
    "stripe_fill", "sketch", "inverted", "noisy_bg",
]


# ── Shape geometry ──


def _regular_polygon(n, phase=0.0):
    """Unit-radius regular polygon vertices."""
    return [
        (math.cos(2 * math.pi * k / n + phase),
         math.sin(2 * math.pi * k / n + phase))
        for k in range(n)
    ]


def _star(n_points=5, inner=0.45):
    verts = []
    for k in range(2 * n_points):
        r = 1.0 if k % 2 == 0 else inner
        a = math.pi * k / n_points - math.pi / 2
        verts.append((r * math.cos(a), r * math.sin(a)))
    return verts


def _cross(arm=0.35):
    a = arm
    return [
        (-a, -1), (a, -1), (a, -a), (1, -a), (1, a), (a, a),
        (a, 1), (-a, 1), (-a, a), (-1, a), (-1, -a), (-a, -a),
    ]


def _arrow():
    return [
        (-1, -0.35), (0.2, -0.35), (0.2, -0.75), (1, 0),
        (0.2, 0.75), (0.2, 0.35), (-1, 0.35),
    ]


def shape_vertices(name):
    """Canonical unit-scale vertex list for a shape (circle → 40-gon)."""
    if name == "triangle":
        return _regular_polygon(3, phase=-math.pi / 2)
    if name == "square":
        return _regular_polygon(4, phase=math.pi / 4)
    if name == "pentagon":
        return _regular_polygon(5, phase=-math.pi / 2)
    if name == "hexagon":
        return _regular_polygon(6)
    if name == "star":
        return _star()
    if name == "circle":
        return _regular_polygon(40)
    if name == "cross":
        return _cross()
    if name == "arrow":
        return _arrow()
    raise ValueError(f"Unknown shape: {name}")


def _place(verts, cx, cy, scale, angle):
    """Rotate, scale, and translate unit vertices onto the canvas."""
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return [
        (cx + scale * (x * cos_a - y * sin_a),
         cy + scale * (x * sin_a + y * cos_a))
        for x, y in verts
    ]


# ── Style renderers ──
# Each takes (polygon vertex list, rng) and returns a PIL RGB image.


def _mask_of(poly, size=CANVAS):
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).polygon(poly, fill=255)
    return mask


def _render_solid_white_bg(poly, rng):
    img = Image.new("RGB", (CANVAS, CANVAS), (245, 245, 245))
    ImageDraw.Draw(img).polygon(poly, fill=(40, 90, 200))
    return img


def _render_outline(poly, rng):
    img = Image.new("RGB", (CANVAS, CANVAS), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    n = len(poly)
    for i in range(n):
        draw.line([poly[i], poly[(i + 1) % n]], fill=(20, 20, 20), width=6)
    return img


def _render_solid_black_bg(poly, rng):
    img = Image.new("RGB", (CANVAS, CANVAS), (15, 15, 15))
    ImageDraw.Draw(img).polygon(poly, fill=(230, 180, 40))
    return img


def _render_stipple_fill(poly, rng):
    img = Image.new("RGB", (CANVAS, CANVAS), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    mask = _mask_of(poly)
    mask_px = mask.load()
    for _ in range(1200):
        x = int(rng.integers(0, CANVAS))
        y = int(rng.integers(0, CANVAS))
        if mask_px[x, y]:
            r = int(rng.integers(2, 5))
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(60, 60, 160))
    return img


def _render_stripe_fill(poly, rng):
    stripes = Image.new("RGB", (CANVAS, CANVAS), (245, 245, 245))
    sdraw = ImageDraw.Draw(stripes)
    for y in range(0, CANVAS, 14):
        sdraw.rectangle([0, y, CANVAS, y + 6], fill=(180, 40, 60))
    img = Image.new("RGB", (CANVAS, CANVAS), (245, 245, 245))
    img.paste(stripes, mask=_mask_of(poly))
    return img


def _render_sketch(poly, rng):
    img = Image.new("RGB", (CANVAS, CANVAS), (250, 248, 240))
    draw = ImageDraw.Draw(img)
    n = len(poly)
    for _ in range(3):  # multiple jittered strokes
        jittered = [
            (x + float(rng.uniform(-4, 4)), y + float(rng.uniform(-4, 4)))
            for x, y in poly
        ]
        for i in range(n):
            draw.line(
                [jittered[i], jittered[(i + 1) % n]],
                fill=(60, 60, 70), width=3,
            )
    return img


def _render_inverted(poly, rng):
    img = Image.new("RGB", (CANVAS, CANVAS), (120, 40, 130))
    ImageDraw.Draw(img).polygon(poly, fill=(250, 250, 250))
    return img


def _render_noisy_bg(poly, rng):
    img = Image.new("RGB", (CANVAS, CANVAS), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    for _ in range(400):  # background clutter dots
        x = int(rng.integers(0, CANVAS))
        y = int(rng.integers(0, CANVAS))
        r = int(rng.integers(2, 6))
        g = int(rng.integers(140, 240))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(g, g, g))
    draw.polygon(poly, fill=(30, 140, 70))
    return img


STYLE_RENDERERS = {
    "solid_white_bg": _render_solid_white_bg,
    "outline": _render_outline,
    "solid_black_bg": _render_solid_black_bg,
    "stipple_fill": _render_stipple_fill,
    "stripe_fill": _render_stripe_fill,
    "sketch": _render_sketch,
    "inverted": _render_inverted,
    "noisy_bg": _render_noisy_bg,
}


class StyleContentGrid(Dataset):
    """64-image grid: every (shape, style) combination, deterministic.

    __getitem__ returns (image, content_idx); style/content indices are
    also available via .content_labels / .style_labels.
    """

    def __init__(self, transform=None, seed=42):
        self.transform = transform
        self.cells = [
            (ci, si)
            for ci in range(len(SHAPE_NAMES))
            for si in range(len(STYLE_NAMES))
        ]
        self.content_labels = np.array([c for c, _ in self.cells])
        self.style_labels = np.array([s for _, s in self.cells])
        self.targets = self.content_labels.tolist()

        # Per-cell deterministic jitter so same-content pairs differ at
        # the pixel level and the model must abstract shape identity.
        rng = np.random.default_rng(seed)
        self._jitter = []
        for _ in self.cells:
            self._jitter.append((
                float(rng.uniform(-math.pi / 12, math.pi / 12)),  # rotation
                float(rng.uniform(-12, 12)),                       # dx
                float(rng.uniform(-12, 12)),                       # dy
                float(rng.uniform(0.85, 1.0)),                     # scale
                int(rng.integers(0, 2**31 - 1)),                   # style seed
            ))

    def __len__(self):
        return len(self.cells)

    def render(self, idx):
        ci, si = self.cells[idx]
        angle, dx, dy, scale_f, style_seed = self._jitter[idx]
        verts = shape_vertices(SHAPE_NAMES[ci])
        poly = _place(
            verts,
            CANVAS / 2 + dx, CANVAS / 2 + dy,
            scale=140 * scale_f, angle=angle,
        )
        rng = np.random.default_rng(style_seed)
        return STYLE_RENDERERS[STYLE_NAMES[si]](poly, rng)

    def __getitem__(self, idx):
        img = self.render(idx)
        if self.transform:
            img = self.transform(img)
        return img, int(self.content_labels[idx])
