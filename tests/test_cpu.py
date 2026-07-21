"""CPU-only tests: scanner math, probes, synthetic confounds, anatomy.

These run without any model download (a tiny dummy transformer stands in),
so the whole logic layer is verifiable locally before burning GPU hours.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from benchmarks.style_content import SHAPE_NAMES, STYLE_NAMES, StyleContentGrid
from benchmarks.synthetic import (
    CountingDataset,
    InsideOutsideDataset,
    SameDifferentDataset,
    SymmetryDataset,
    _dist_to_polygon_boundary,
    _point_in_polygon,
    _polygon_area,
)
from probes.nearest_centroid import (
    compute_centroids,
    score_nearest_centroid,
    score_nearest_centroid_detailed,
    select_borderline_images,
)
from scanner.anatomy import centered_similarity_curves, pair_categories
from scanner.layer_loop import RYSScanner


# ── Dummy model ──


class DummyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x, attention_mask=None, causal_attention_mask=None):
        # Tuple return + extra positional args, mimicking EVA-CLIP layers
        return (x + torch.tanh(self.fc(x)),)


class DummyEncoder(nn.Module):
    def __init__(self, dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(DummyBlock(dim) for _ in range(n_layers))


class DummyViT(nn.Module):
    def __init__(self, dim=16, n_layers=6, n_tokens=5):
        super().__init__()
        self.proj = nn.Linear(3, dim)
        self.encoder = DummyEncoder(dim, n_layers)
        self.n_tokens = n_tokens

    def embeddings(self, pixel_values):
        # (B, 3, H, W) -> (B, n_tokens, dim), deterministic in the input
        b = pixel_values.shape[0]
        pooled = pixel_values.mean(dim=(2, 3))  # (B, 3)
        base = self.proj(pooled)  # (B, dim)
        tokens = torch.stack(
            [base * (k + 1) / self.n_tokens for k in range(self.n_tokens)],
            dim=1,
        )
        return tokens


@pytest.fixture(scope="module")
def scanner():
    torch.manual_seed(0)
    model = DummyViT()
    return RYSScanner(
        model, device="cpu",
        layer_fn=lambda layer, h: layer(h, None, None),
    )


@pytest.fixture(scope="module")
def sample_input():
    torch.manual_seed(1)
    return torch.randn(2, 3, 8, 8)


class TestScanner:
    def test_num_configs(self, scanner):
        n = scanner.num_layers
        assert n == 6
        assert scanner.num_configs() == n * (n + 1) // 2
        assert len(list(scanner.all_configs())) == scanner.num_configs()

    def test_invalid_configs_raise(self, scanner, sample_input):
        with pytest.raises(ValueError):
            scanner.run_config(3, 3, sample_input)
        with pytest.raises(ValueError):
            scanner.run_config(4, 2, sample_input)
        cached = scanner.cache_baseline_states(sample_input)
        with pytest.raises(ValueError):
            scanner.run_config_from_cache(4, 2, cached)
        with pytest.raises(ValueError):
            scanner.run_config_from_cache(1, 3, cached, repeats=1)

    def test_cache_matches_direct(self, scanner, sample_input):
        baseline = scanner.get_baseline_embeddings(sample_input)
        cached = scanner.cache_baseline_states(sample_input)
        assert len(cached) == scanner.num_layers + 1
        assert torch.allclose(baseline, scanner._pool_fn(cached[-1]), atol=1e-6)

        direct = scanner.run_config(1, 4, sample_input)
        from_cache = scanner.run_config_from_cache(1, 4, cached)
        assert torch.allclose(direct, from_cache, atol=1e-6)

    def test_duplication_changes_output(self, scanner, sample_input):
        baseline = scanner.get_baseline_embeddings(sample_input)
        modified = scanner.run_config(1, 4, sample_input)
        assert not torch.allclose(baseline, modified, atol=1e-4)

    def test_repeats_semantics(self, scanner, sample_input):
        """repeats=k must equal manually running the block k times."""
        i, j = 2, 4
        cached = scanner.cache_baseline_states(sample_input)

        for repeats in (2, 3):
            expected = cached[j]
            for _ in range(repeats - 1):
                for idx in range(i, j):
                    expected = scanner._run_layer(idx, expected)
            for idx in range(j, scanner.num_layers):
                expected = scanner._run_layer(idx, expected)
            got = scanner.run_config_from_cache(i, j, cached, repeats=repeats)
            assert torch.allclose(got, scanner._pool_fn(expected).clone(), atol=1e-6)

        r2 = scanner.run_config_from_cache(i, j, cached, repeats=2)
        r3 = scanner.run_config_from_cache(i, j, cached, repeats=3)
        assert not torch.allclose(r2, r3, atol=1e-5)

    def test_full_block_duplication(self, scanner, sample_input):
        """(0, L) duplicates every layer; must run without index errors."""
        out = scanner.run_config(0, scanner.num_layers, sample_input)
        assert out.shape[0] == sample_input.shape[0]

    def test_single_layer_repeat_configs(self, scanner):
        cfgs = list(scanner.single_layer_repeat_configs(k_values=(3, 4)))
        assert len(cfgs) == 2 * scanner.num_layers
        assert all(j == i + 1 and k in (3, 4) for i, j, k in cfgs)


# ── Probes ──


class TestNearestCentroid:
    def _toy(self):
        rng = np.random.default_rng(0)
        centroids_true = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])}
        feats, labels = [], []
        for lbl, c in centroids_true.items():
            feats.append(c + rng.normal(0, 0.05, size=(20, 2)))
            labels.extend([lbl] * 20)
        return np.concatenate(feats), np.array(labels)

    def test_perfect_separation(self):
        feats, labels = self._toy()
        cents = compute_centroids(feats, labels)
        s = score_nearest_centroid(feats, labels, cents)
        assert s["accuracy"] == 1.0
        assert s["mrr"] == 1.0

    def test_detailed_consistency(self):
        feats, labels = self._toy()
        cents = compute_centroids(feats, labels)
        d = score_nearest_centroid_detailed(feats, labels, cents)
        assert d["correct"].all()
        assert (d["margin"] > 0).all()
        assert (d["rank"] == 1).all()

    def test_missing_class_raises(self):
        feats, labels = self._toy()
        cents = compute_centroids(feats[:20], labels[:20])  # only class 0
        with pytest.raises(ValueError):
            score_nearest_centroid(feats, labels, cents)

    def test_borderline_selection_is_stratified(self):
        rng = np.random.default_rng(3)
        n = 400
        feats = rng.normal(size=(n, 8))
        labels = rng.integers(0, 4, size=n)
        cents = compute_centroids(feats[:100], labels[:100])
        sel = select_borderline_images(feats, labels, cents, n_select=100)
        assert len(sel) == 100
        assert len(set(sel)) == 100
        d = score_nearest_centroid_detailed(feats[sel], labels[sel], cents)
        # Stratified around the boundary: baseline is near 50%, not near 0
        assert 0.3 <= d["correct"].mean() <= 0.7


# ── Synthetic dataset confound checks ──


class TestSyntheticConfounds:
    def test_counting_no_overlap(self):
        ds = CountingDataset(n_per_class=5, seed=7)
        for params, target in zip(ds._params, ds.targets):
            _bg, dots = params
            assert len(dots) == target + 1
            for a in range(len(dots)):
                for b in range(a + 1, len(dots)):
                    x1, y1, r1, _ = dots[a]
                    x2, y2, r2, _ = dots[b]
                    assert math.hypot(x1 - x2, y1 - y2) > r1 + r2, \
                        "dots overlap — visible count would be below label"

    def test_same_different_area_matched(self):
        ds = SameDifferentDataset(n_per_class=50, seed=7)
        for params, target in zip(ds._params, ds.targets):
            _bg, lv, _lc, rv, _rc = params
            if target == 0:  # different
                ratio = _polygon_area(rv) / _polygon_area(lv)
                assert abs(ratio - 1.0) < 0.01, \
                    "area mismatch is a same/different shortcut"

    def test_symmetry_half_marginals_match(self):
        ds = SymmetryDataset(n_per_class=50, seed=7)
        for params, target in zip(ds._params, ds.targets):
            _bg, left, right = params
            # (y, r, color) multisets must match between halves for BOTH
            # classes, so color/size histograms are never diagnostic.
            left_sig = sorted((y, r, c) for _x, y, r, c in left)
            right_sig = sorted((y, r, c) for _x, y, r, c in right)
            assert left_sig == right_sig

    def test_inside_outside_margins(self):
        ds = InsideOutsideDataset(n_per_class=50, seed=7)
        bad_margin = 0
        for params, target in zip(ds._params, ds.targets):
            _bg, contour, dx, dy, dot_r = params
            assert _point_in_polygon(dx, dy, contour) == bool(target)
            if _dist_to_polygon_boundary(dx, dy, contour) < dot_r + 4:
                bad_margin += 1
        assert bad_margin == 0, "ambiguous dot placements near the contour"

    def test_renders(self):
        for ds in (CountingDataset(n_per_class=1, seed=1),
                   SameDifferentDataset(n_per_class=1, seed=1),
                   SymmetryDataset(n_per_class=1, seed=1),
                   InsideOutsideDataset(n_per_class=1, seed=1)):
            img, label = ds[0]
            assert img.size == (448, 448)


# ── Style/content grid + anatomy ──


class TestAnatomy:
    def test_grid_structure(self):
        grid = StyleContentGrid(n_instances=1)
        assert len(grid) == len(SHAPE_NAMES) * len(STYLE_NAMES) == 64
        img, label = grid[0]
        assert img.size == (448, 448)
        assert label == grid.content_labels[0]
        assert len(StyleContentGrid()) == 128  # default: 2 instances/cell

    def test_pair_categories_counts(self):
        grid = StyleContentGrid(n_instances=1)
        _iu, _ju, cat = pair_categories(grid.content_labels, grid.style_labels)
        # 8x8 grid, 1/cell: C(8,2)*8 = 224 same-content and 224 same-style
        assert (cat == 0).sum() == 224
        assert (cat == 1).sum() == 224
        assert (cat == 2).sum() == 1568
        assert (cat == 3).sum() == 0

        grid2 = StyleContentGrid()  # 2 instances/cell, n=128
        _iu, _ju, cat2 = pair_categories(grid2.content_labels, grid2.style_labels)
        # per content: C(16,2)=120 pairs, minus 8 same-style-same-content
        assert (cat2 == 0).sum() == 8 * (120 - 8)
        assert (cat2 == 1).sum() == 8 * (120 - 8)
        assert (cat2 == 3).sum() == 64
        assert (cat2 == 2).sum() == 128 * 127 // 2 - 2 * 896 - 64

    def test_same_both_curve_present(self):
        grid = StyleContentGrid()
        rng = np.random.default_rng(0)
        feats = rng.normal(size=(1, len(grid), 8))
        curves = centered_similarity_curves(
            feats, grid.content_labels, grid.style_labels
        )
        assert "same_both" in curves and len(curves["same_both"]) == 1

    def test_curves_detect_planted_structure(self):
        """Features organized by content must yield content > style."""
        grid = StyleContentGrid()
        rng = np.random.default_rng(0)
        content_dirs = rng.normal(size=(8, 32))
        feats_content = np.stack([
            content_dirs[c] + rng.normal(0, 0.1, 32)
            for c in grid.content_labels
        ])
        curves = centered_similarity_curves(
            feats_content[None], grid.content_labels, grid.style_labels
        )
        assert curves["same_content"][0] > curves["same_style"][0]
        assert curves["reasoning_layers"] == [0]

        style_dirs = rng.normal(size=(8, 32))
        feats_style = np.stack([
            style_dirs[s] + rng.normal(0, 0.1, 32)
            for s in grid.style_labels
        ])
        curves = centered_similarity_curves(
            feats_style[None], grid.content_labels, grid.style_labels
        )
        assert curves["same_style"][0] > curves["same_content"][0]
        assert curves["reasoning_layers"] == []
