"""GPU integration tests for the RYS scanner against the real model.

Requires the model download (~73 GB for eva18b) and a GPU — run on the
training box, not locally. The logic-level tests live in test_cpu.py and
run anywhere.

    pytest tests/test_layer_loop.py -v --model eva18b
"""

import numpy as np
import pytest
import torch
from PIL import Image


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU + model download"
)


@pytest.fixture(scope="module")
def loaded(request):
    from scripts.model_registry import load_model
    key = request.config.getoption("--model", default="eva18b")
    return load_model(key, device="cuda")


@pytest.fixture(scope="module")
def sample_input(loaded):
    _scanner, processor, spec = loaded
    img = Image.fromarray(
        np.random.randint(0, 255, (spec["image_size"], spec["image_size"], 3),
                          dtype=np.uint8)
    )
    return processor(images=img, return_tensors="pt")["pixel_values"]


@requires_gpu
class TestRealModel:
    def test_layer_count(self, loaded):
        scanner, _, spec = loaded
        assert scanner.num_layers == spec["expected_layers"]

    def test_baseline_shape(self, loaded, sample_input):
        scanner, _, spec = loaded
        emb = scanner.get_baseline_embeddings(sample_input)
        assert emb.shape == (1, spec["expected_hidden"])

    def test_cache_consistency(self, loaded, sample_input):
        scanner, _, _ = loaded
        baseline = scanner.get_baseline_embeddings(sample_input)
        cached = scanner.cache_baseline_states(sample_input)
        assert len(cached) == scanner.num_layers + 1
        assert torch.allclose(
            baseline.float(), scanner._pool_fn(cached[-1]).float(), atol=1e-4
        )

    def test_cached_config_matches_direct(self, loaded, sample_input):
        scanner, _, _ = loaded
        L = scanner.num_layers
        i, j = L // 4, L // 2
        direct = scanner.run_config(i, j, sample_input)
        cached = scanner.cache_baseline_states(sample_input)
        from_cache = scanner.run_config_from_cache(i, j, cached)
        assert torch.allclose(direct.float(), from_cache.float(), atol=1e-4)

    def test_duplication_changes_output(self, loaded, sample_input):
        scanner, _, _ = loaded
        L = scanner.num_layers
        baseline = scanner.get_baseline_embeddings(sample_input)
        modified = scanner.run_config(L // 4, L // 2, sample_input)
        assert not torch.allclose(baseline.float(), modified.float(), atol=1e-3)
