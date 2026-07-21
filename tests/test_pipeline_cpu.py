"""End-to-end pipeline test on CPU with a dummy model.

Runs the real run_scan.main() (synthetic datasets only), then heatmap
generation, the report, resume, and the stage-2 confirmation — verifying
the entire orchestration without a GPU or model download.
"""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scanner.layer_loop import RYSScanner
from tests.test_cpu import DummyViT


def fake_load_model(key, device="cpu", dtype=torch.float32):
    torch.manual_seed(0)
    model = DummyViT(dim=16, n_layers=6)
    scanner = RYSScanner(
        model, device="cpu",
        layer_fn=lambda layer, h: layer(h, None, None),
    )
    processor = SimpleNamespace(
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    )
    spec = {"name": "dummy", "image_size": 64,
            "expected_layers": 6, "expected_hidden": 16}
    return scanner, processor, spec


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("pipeline")


SCAN_ARGS = [
    "run_scan.py",
    "--datasets", "counting", "same_different",
    "--n-candidates", "120",
    "--n-hard", "24", "--n-rand", "24",
    "--ref-budget", "20", "--n-per-class", "2",
    "--batch-size", "16", "--num-workers", "0",
    "--config-batch-size", "12",
    "--repeat-scan",
]


def run_scan_with(monkeypatch, argv):
    import scripts.run_scan as run_scan
    monkeypatch.setattr(run_scan, "load_model", fake_load_model)
    monkeypatch.setattr(sys, "argv", argv)
    run_scan.main()


def test_full_pipeline(monkeypatch, workdir):
    results = str(workdir / "results")
    outputs = str(workdir / "outputs")
    data = str(workdir / "data")

    argv = SCAN_ARGS + ["--data-dir", data, "--output-dir", results]
    run_scan_with(monkeypatch, argv)

    # 6 layers -> 21 configs; matrices complete for both splits
    for ds in ("counting", "same_different"):
        for split in ("hard", "rand"):
            m = np.load(os.path.join(results, f"{ds}_{split}_accuracy_matrix.npy"))
            assert m.shape == (6, 7)
            valid = np.array([[i < j for j in range(7)] for i in range(6)])
            assert not np.any(np.isnan(m[valid]))
            assert np.all(np.isnan(m[~valid]))
        z = np.load(os.path.join(results, f"{ds}_percase.npz"))
        assert z["hard_correct"].shape == (21, 24)
        assert (z["hard_correct"] >= 0).all()
        with open(os.path.join(results, f"{ds}_repeats.json")) as f:
            repeats = json.load(f)
        assert len(repeats) == 12  # 6 layers x k in (3, 4)

    assert os.path.exists(os.path.join(results, "probe_sets.json"))
    assert os.path.exists(
        os.path.join(results, "aggregated_rand_accuracy_zscore.npy"))

    # Noise control recorded
    with open(os.path.join(results, "probe_meta.json")) as f:
        meta = json.load(f)
    assert "noise_control" in meta["counting"]

    # Resume: must be a fast no-op that leaves results identical
    before = np.load(os.path.join(results, "counting_rand_accuracy_matrix.npy"))
    run_scan_with(monkeypatch, argv + ["--resume"])
    after = np.load(os.path.join(results, "counting_rand_accuracy_matrix.npy"))
    np.testing.assert_array_equal(before, after)

    # Heatmaps
    from visualization.heatmap import generate_all_heatmaps
    generate_all_heatmaps(results, outputs, top_k=3)
    assert os.path.exists(
        os.path.join(outputs, "heatmap_counting_rand_accuracy.png"))
    assert os.path.exists(
        os.path.join(outputs, "heatmap_aggregated_rand_accuracy_zscore.png"))
    assert os.path.exists(os.path.join(outputs, "pareto_rand_accuracy.png"))
    assert os.path.exists(os.path.join(outputs, "repeats_counting.png"))

    # Report with bootstrap CIs
    from scripts.report import generate_report
    report_path = os.path.join(outputs, "report.md")
    generate_report(results, report_path, top_k=5)
    text = open(report_path).read()
    assert "Top 5 Configs" in text
    assert "95% CI" in text
    assert "noise" in text.lower()

    # Stage-2 confirmation
    import scripts.confirm_top as confirm_top
    monkeypatch.setattr(confirm_top, "load_model", fake_load_model)
    monkeypatch.setattr(sys, "argv", [
        "confirm_top.py", "--data-dir", data, "--results-dir", results,
        "--datasets", "counting", "same_different",
        "--top-k", "3", "--n-random-configs", "3",
        "--n-test", "40", "--batch-size", "16", "--num-workers", "0",
    ])
    confirm_top.main()
    with open(os.path.join(results, "confirm_results.json")) as f:
        confirm = json.load(f)
    assert len(confirm["per_config"]) == 6
    for row in confirm["per_config"]:
        assert row["ci_lo"] <= row["delta_acc"] <= row["ci_hi"]


def test_anatomy_pipeline(monkeypatch, workdir):
    import scripts.run_anatomy as run_anatomy
    outdir = str(workdir / "anatomy")
    monkeypatch.setattr(run_anatomy, "load_model", fake_load_model)
    monkeypatch.setattr(sys, "argv", [
        "run_anatomy.py", "--output-dir", outdir, "--batch-size", "16",
    ])
    run_anatomy.main()
    with open(os.path.join(outdir, "anatomy.json")) as f:
        anatomy = json.load(f)
    assert "mean_patch" in anatomy and "cls" in anatomy
    assert len(anatomy["mean_patch"]["same_content"]) == 7  # 6 layers + emb
    assert os.path.exists(os.path.join(outdir, "anatomy_mean_patch.png"))
    assert os.path.exists(os.path.join(outdir, "anatomy_pca_cls.png"))
