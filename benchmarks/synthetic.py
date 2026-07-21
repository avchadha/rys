"""Synthetic visual reasoning probes for RYS experiments.

Each dataset isolates a specific visual cognitive operation, generated
procedurally so no download is needed. Confounds (total area, brightness,
element count) are controlled where possible.

Probes:
  - Counting:         Numerosity perception (how many dots?)
  - SameDifferent:    Relational identity (are these two shapes the same?)
  - SpatialRelation:  Spatial reasoning (where is the blue shape relative to red?)
  - Symmetry:         Global structure perception (is this dot pattern symmetric?)
  - InsideOutside:    Topological reasoning (is the dot inside the contour?)
"""

import math

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


# ── Helpers ──


def _random_color(rng):
    """Generate a random saturated color (avoids near-gray)."""
    color = rng.integers(30, 230, size=3)
    # Boost saturation: push one channel high, one low
    order = rng.permutation(3)
    color[order[0]] = min(230, color[order[0]] + 60)
    color[order[2]] = max(30, color[order[2]] - 60)
    return tuple(color.tolist())


def _random_polygon_vertices(rng, cx, cy, radius, n_vertices):
    """Generate vertices of a random simple polygon centered at (cx, cy).

    Vertices are generated in sorted angular order (star-shaped from center)
    to guarantee the polygon is non-self-intersecting.
    """
    angles = np.sort(rng.uniform(0, 2 * math.pi, size=n_vertices))
    radii = radius * rng.uniform(0.6, 1.0, size=n_vertices)
    vertices = [
        (float(cx + r * math.cos(a)), float(cy + r * math.sin(a)))
        for a, r in zip(angles, radii)
    ]
    return vertices


def _polygon_area(polygon):
    """Shoelace formula."""
    area = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _dist_to_polygon_boundary(px, py, polygon):
    """Minimum distance from a point to any polygon edge."""
    best = float("inf")
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq == 0:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
        cx, cy = x1 + t * dx, y1 + t * dy
        best = min(best, math.hypot(px - cx, py - cy))
    return best


def _point_in_polygon(px, py, polygon):
    """Ray-casting algorithm for point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


# ── Counting ──


class CountingDataset(Dataset):
    """Count randomly placed dots (1-10). 10-class classification.

    Confound control: dot sizes are randomized so total filled area
    does not correlate perfectly with count. Background color varies.
    """

    N_CLASSES = 10  # counts 1 through 10

    def __init__(self, n_per_class=500, transform=None, seed=42):
        self.transform = transform
        self.size = 448

        rng = np.random.default_rng(seed)
        self._params = []
        self.targets = []

        for count in range(1, self.N_CLASSES + 1):
            for _ in range(n_per_class):
                self._params.append(self._gen_params(rng, count))
                self.targets.append(count - 1)

    def _gen_params(self, rng, count):
        s = self.size
        bg = _random_color(rng)

        # Dots must never overlap or touch (merged blobs would make the
        # visible count smaller than the label). Radii are drawn FIRST,
        # independently of placement success, so the radius distribution
        # does not shift with count; only positions are rejection-sampled,
        # and on failure the whole layout restarts with fresh radii.
        for _layout_attempt in range(100):
            radii = [int(rng.integers(12, 40)) for _ in range(count)]
            dots = []
            ok = True
            for r in sorted(radii, reverse=True):  # place big dots first
                for _attempt in range(300):
                    x = int(rng.integers(r + 5, s - r - 5))
                    y = int(rng.integers(r + 5, s - r - 5))
                    if all(
                        (x - ox) ** 2 + (y - oy) ** 2 > (r + orad + 6) ** 2
                        for ox, oy, orad, _ in dots
                    ):
                        dots.append((x, y, r, _random_color(rng)))
                        break
                else:
                    ok = False
                    break
            if ok:
                return (bg, dots)

        raise RuntimeError(
            f"CountingDataset: could not place {count} non-overlapping dots "
            "after 100 layout attempts."
        )

    def _render(self, params):
        bg, dots = params
        img = Image.new("RGB", (self.size, self.size), bg)
        draw = ImageDraw.Draw(img)
        for x, y, r, color in dots:
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
        return img

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self._render(self._params[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]


# ── Same / Different ──


class SameDifferentDataset(Dataset):
    """Two shapes side by side — are they the same shape? Binary classification.

    Confound control: colors differ between left and right regardless of
    same/different.  'Same' pairs use a slight rotation so the model must
    do abstract shape comparison, not pixel matching.  'Different' pairs
    use the same vertex count as the left shape.
    """

    N_CLASSES = 2  # 0=different, 1=same

    def __init__(self, n_per_class=2500, transform=None, seed=42):
        self.transform = transform
        self.size = 448

        rng = np.random.default_rng(seed)
        self._params = []
        self.targets = []

        for label in range(self.N_CLASSES):
            for _ in range(n_per_class):
                self._params.append(self._gen_params(rng, same=(label == 1)))
                self.targets.append(label)

    def _gen_params(self, rng, same):
        s = self.size
        bg = (220, 220, 220)
        radius = int(rng.integers(45, 85))
        n_verts = int(rng.integers(3, 9))

        # Left shape
        left_verts = _random_polygon_vertices(
            rng, s // 4, s // 2, radius, n_verts
        )
        left_color = _random_color(rng)

        # Right shape
        right_color = _random_color(rng)
        if same:
            # Same shape, translated + rotated (wide enough that pixel/template
            # matching fails and abstract shape comparison is required)
            angle = float(rng.uniform(-0.6, 0.6))
            lcx = sum(v[0] for v in left_verts) / len(left_verts)
            lcy = sum(v[1] for v in left_verts) / len(left_verts)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            cx2, cy2 = s * 3 / 4, s / 2
            right_verts = [
                (cx2 + (vx - lcx) * cos_a - (vy - lcy) * sin_a,
                 cy2 + (vx - lcx) * sin_a + (vy - lcy) * cos_a)
                for vx, vy in left_verts
            ]
        else:
            # Different shape, same vertex count — rescaled around its center
            # to match the left shape's AREA, so total filled area is not a
            # shortcut for same/different.
            right_verts = _random_polygon_vertices(
                rng, s * 3 // 4, s // 2, radius, n_verts
            )
            area_l = _polygon_area(left_verts)
            area_r = _polygon_area(right_verts)
            if area_r > 1e-6:
                k = math.sqrt(area_l / area_r)
                rcx = sum(v[0] for v in right_verts) / len(right_verts)
                rcy = sum(v[1] for v in right_verts) / len(right_verts)
                right_verts = [
                    (rcx + (vx - rcx) * k, rcy + (vy - rcy) * k)
                    for vx, vy in right_verts
                ]

        return (bg, left_verts, left_color, right_verts, right_color)

    def _render(self, params):
        bg, left_verts, left_color, right_verts, right_color = params
        img = Image.new("RGB", (self.size, self.size), bg)
        draw = ImageDraw.Draw(img)
        draw.polygon(left_verts, fill=left_color)
        draw.polygon(right_verts, fill=right_color)
        return img

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self._render(self._params[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]


# ── Spatial Relation ──


class SpatialRelationDataset(Dataset):
    """Two shapes — classify their spatial relation. 4-class classification.

    A red circle (reference) and blue triangle (target) are placed so that
    the target is above / below / left of / right of the reference.

    Confound control: reference position is jittered so target position
    alone does not determine the label.
    """

    N_CLASSES = 4  # 0=above, 1=below, 2=left, 3=right
    RELATIONS = ["above", "below", "left", "right"]

    def __init__(self, n_per_class=1250, transform=None, seed=42):
        self.transform = transform
        self.size = 448

        rng = np.random.default_rng(seed)
        self._params = []
        self.targets = []

        for label in range(self.N_CLASSES):
            for _ in range(n_per_class):
                self._params.append(self._gen_params(rng, label))
                self.targets.append(label)

    def _gen_params(self, rng, relation):
        s = self.size
        bg = (240, 240, 240)
        ref_r = int(rng.integers(25, 50))
        tgt_r = int(rng.integers(25, 50))

        # Reference near center with jitter
        ref_x = s // 2 + int(rng.integers(-70, 70))
        ref_y = s // 2 + int(rng.integers(-70, 70))

        # Target placed relative to reference
        gap = int(rng.integers(80, 160))
        cross_jitter = int(rng.integers(-25, 25))
        if relation == 0:  # target above reference
            tgt_x, tgt_y = ref_x + cross_jitter, ref_y - gap
        elif relation == 1:  # target below reference
            tgt_x, tgt_y = ref_x + cross_jitter, ref_y + gap
        elif relation == 2:  # target left of reference
            tgt_x, tgt_y = ref_x - gap, ref_y + cross_jitter
        else:  # target right of reference
            tgt_x, tgt_y = ref_x + gap, ref_y + cross_jitter

        # Clamp to canvas
        ref_x = max(ref_r + 5, min(s - ref_r - 5, ref_x))
        ref_y = max(ref_r + 5, min(s - ref_r - 5, ref_y))
        tgt_x = max(tgt_r + 5, min(s - tgt_r - 5, tgt_x))
        tgt_y = max(tgt_r + 5, min(s - tgt_r - 5, tgt_y))

        # Triangle vertices
        tgt_verts = [
            (float(tgt_x), float(tgt_y - tgt_r)),
            (float(tgt_x - tgt_r), float(tgt_y + tgt_r)),
            (float(tgt_x + tgt_r), float(tgt_y + tgt_r)),
        ]

        return (bg, ref_x, ref_y, ref_r, tgt_verts)

    def _render(self, params):
        bg, ref_x, ref_y, ref_r, tgt_verts = params
        img = Image.new("RGB", (self.size, self.size), bg)
        draw = ImageDraw.Draw(img)
        draw.ellipse(
            [ref_x - ref_r, ref_y - ref_r, ref_x + ref_r, ref_y + ref_r],
            fill=(200, 50, 50),
        )
        draw.polygon(tgt_verts, fill=(50, 50, 200))
        return img

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self._render(self._params[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]


# ── Symmetry ──


class SymmetryDataset(Dataset):
    """Random dot patterns — is the pattern vertically symmetric? Binary.

    Confound control: asymmetric images have the same number of dots on
    each half as symmetric ones, controlling for density and total area.
    """

    N_CLASSES = 2  # 0=asymmetric, 1=symmetric

    def __init__(self, n_per_class=2500, transform=None, seed=42):
        self.transform = transform
        self.size = 448

        rng = np.random.default_rng(seed)
        self._params = []
        self.targets = []

        for label in range(self.N_CLASSES):
            for _ in range(n_per_class):
                self._params.append(
                    self._gen_params(rng, symmetric=(label == 1))
                )
                self.targets.append(label)

    def _gen_params(self, rng, symmetric):
        s = self.size
        bg = (30, 30, 30)
        n_dots = int(rng.integers(10, 25))

        # Left-half dots
        left_dots = []
        for _ in range(n_dots):
            x = int(rng.integers(15, s // 2 - 10))
            y = int(rng.integers(15, s - 15))
            r = int(rng.integers(8, 22))
            color = _random_color(rng)
            left_dots.append((x, y, r, color))

        if symmetric:
            right_dots = [(s - x, y, r, color) for x, y, r, color in left_dots]
        else:
            # Same (y, radius, color) multiset as the mirrored left half —
            # only the x-positions are resampled. This matches the two
            # halves' color histograms, dot sizes, and vertical marginals,
            # so only true spatial mirroring distinguishes the classes.
            right_dots = [
                (int(rng.integers(s // 2 + 10, s - 15)), y, r, color)
                for _, y, r, color in left_dots
            ]

        return (bg, left_dots, right_dots)

    def _render(self, params):
        bg, left_dots, right_dots = params
        img = Image.new("RGB", (self.size, self.size), bg)
        draw = ImageDraw.Draw(img)
        for dots in [left_dots, right_dots]:
            for x, y, r, color in dots:
                draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
        return img

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self._render(self._params[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]


# ── Inside / Outside ──


class InsideOutsideDataset(Dataset):
    """A closed contour and a red dot — is the dot inside? Binary.

    A large irregular polygon is drawn as a contour outline. A red dot
    is placed either inside or outside. Uses ray-casting for ground truth.
    """

    N_CLASSES = 2  # 0=outside, 1=inside

    def __init__(self, n_per_class=2500, transform=None, seed=42):
        self.transform = transform
        self.size = 448

        rng = np.random.default_rng(seed)
        self._params = []
        self.targets = []

        for label in range(self.N_CLASSES):
            for _ in range(n_per_class):
                self._params.append(
                    self._gen_params(rng, inside=(label == 1))
                )
                self.targets.append(label)

    def _gen_params(self, rng, inside):
        s = self.size
        bg = (240, 240, 240)
        dot_r = 12
        margin = dot_r + 4 + 8  # dot radius + line half-width + slack

        # A randomly generated polygon can be a thin sliver whose interior
        # has NO margin-safe region — regenerate the contour until the dot
        # can be placed. Placement constraints beyond the class label:
        #   1. Boundary margin: the dot must clear the contour line by a
        #      visible margin, so the label is never visually ambiguous
        #      (esp. after 448 -> 224 downscaling).
        #   2. Annulus band: the dot's distance from the polygon center
        #      must lie in a band where both classes CAN occur (the contour
        #      radius spans 0.6-1.0 x contour_r). This WEAKENS the
        #      distance-from-center shortcut but does not eliminate it:
        #      inside dots concentrate below ~1.0R, outside dots above —
        #      interpret this probe's results with that caveat.
        for _contour_attempt in range(50):
            cx = s // 2 + int(rng.integers(-30, 30))
            cy = s // 2 + int(rng.integers(-30, 30))
            n_verts = int(rng.integers(6, 12))
            contour_r = int(rng.integers(100, 160))
            contour = _random_polygon_vertices(rng, cx, cy, contour_r, n_verts)
            band_lo, band_hi = 0.45 * contour_r, 1.25 * contour_r

            for _ in range(1000):
                dx = int(rng.integers(dot_r + 10, s - dot_r - 10))
                dy = int(rng.integers(dot_r + 10, s - dot_r - 10))
                d_center = math.hypot(dx - cx, dy - cy)
                if not (band_lo <= d_center <= band_hi):
                    continue
                if _dist_to_polygon_boundary(dx, dy, contour) < margin:
                    continue
                if _point_in_polygon(dx, dy, contour) == inside:
                    return (bg, contour, dx, dy, dot_r)

            # Annulus may be infeasible for this contour; keep the margin
            # and label constraints and try without the band.
            for _ in range(1000):
                dx = int(rng.integers(dot_r + 10, s - dot_r - 10))
                dy = int(rng.integers(dot_r + 10, s - dot_r - 10))
                if (_dist_to_polygon_boundary(dx, dy, contour) >= margin
                        and _point_in_polygon(dx, dy, contour) == inside):
                    return (bg, contour, dx, dy, dot_r)

        raise RuntimeError(
            "InsideOutsideDataset: could not place a valid dot after 50 "
            "contour regenerations — check generation parameters."
        )

    def _render(self, params):
        bg, contour, dx, dy, dot_r = params
        img = Image.new("RGB", (self.size, self.size), bg)
        draw = ImageDraw.Draw(img)
        # Draw contour as thick outline (no fill)
        n = len(contour)
        for i in range(n):
            x1, y1 = contour[i]
            x2, y2 = contour[(i + 1) % n]
            draw.line([(x1, y1), (x2, y2)], fill=(50, 50, 50), width=4)
        # Draw dot
        draw.ellipse(
            [dx - dot_r, dy - dot_r, dx + dot_r, dy + dot_r],
            fill=(200, 40, 40),
        )
        return img

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self._render(self._params[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]
