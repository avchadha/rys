"""Tests for the RYS layer duplication scanner."""

import gc

import torch
import pytest
from transformers import AutoModel, CLIPImageProcessor
from PIL import Image
import numpy as np

from scanner.layer_loop import RYSScanner


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def vision_model():
    full_model = AutoModel.from_pretrained(
        "BAAI/EVA-CLIP-18B",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    vm = full_model.vision_model
    del full_model
    gc.collect()
    return vm


@pytest.fixture(scope="module")
def scanner(vision_model, device):
    return RYSScanner(vision_model, device=device)


@pytest.fixture(scope="module")
def processor():
    return CLIPImageProcessor.from_pretrained("BAAI/EVA-CLIP-18B")


@pytest.fixture(scope="module")
def sample_input(processor, device):
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    inputs = processor(images=img, return_tensors="pt")
    return inputs["pixel_values"].to(device)


class TestRYSScanner:
    def test_num_layers(self, scanner):
        assert scanner.num_layers == 48

    def test_num_configs(self, scanner):
        assert scanner.num_configs() == 1176

    def test_all_configs_count(self, scanner):
        configs = list(scanner.all_configs())
        assert len(configs) == scanner.num_configs()

    def test_all_configs_valid(self, scanner):
        for i, j in scanner.all_configs():
            assert 0 <= i < j <= scanner.num_layers

    def test_validate_config(self, scanner):
        assert scanner.validate_config(0, 1)
        assert scanner.validate_config(0, 48)
        assert scanner.validate_config(10, 20)
        assert not scanner.validate_config(5, 5)
        assert not scanner.validate_config(10, 5)
        assert not scanner.validate_config(-1, 5)
        assert not scanner.validate_config(0, 49)

    def test_invalid_config_raises(self, scanner, sample_input):
        with pytest.raises(ValueError):
            scanner.run_config(5, 5, sample_input)
        with pytest.raises(ValueError):
            scanner.run_config(10, 5, sample_input)

    def test_baseline_embedding_shape(self, scanner, sample_input):
        emb = scanner.get_baseline_embeddings(sample_input)
        assert emb.shape == (1, 5120)

    def test_config_embedding_shape(self, scanner, sample_input):
        emb = scanner.run_config(0, 1, sample_input)
        assert emb.shape == (1, 5120)

    def test_config_differs_from_baseline(self, scanner, sample_input):
        """A non-trivial duplication should change the output."""
        baseline = scanner.get_baseline_embeddings(sample_input)
        modified = scanner.run_config(10, 30, sample_input)
        assert not torch.allclose(baseline.float(), modified.float(), atol=1e-3)

    def test_different_configs_differ(self, scanner, sample_input):
        """Different duplication configs should produce different embeddings."""
        emb1 = scanner.run_config(0, 10, sample_input)
        emb2 = scanner.run_config(20, 40, sample_input)
        assert not torch.allclose(emb1.float(), emb2.float(), atol=1e-3)


class TestCachedForward:
    def test_cache_length(self, scanner, sample_input):
        cached = scanner.cache_baseline_states(sample_input)
        # num_layers + 1 entries: embeddings output + after each layer
        assert len(cached) == scanner.num_layers + 1

    def test_cache_shapes(self, scanner, sample_input):
        cached = scanner.cache_baseline_states(sample_input)
        for state in cached:
            assert state.shape[0] == 1  # batch size
            assert state.shape[2] == 5120  # hidden dim

    def test_cached_baseline_matches_direct(self, scanner, sample_input):
        """CLS from cache should match direct baseline."""
        baseline = scanner.get_baseline_embeddings(sample_input)
        cached = scanner.cache_baseline_states(sample_input)
        cached_cls = cached[-1][:, 0]
        assert torch.allclose(baseline.float(), cached_cls.float(), atol=1e-4)

    def test_cached_config_matches_direct(self, scanner, sample_input):
        """Config run from cache should match direct run_config."""
        i, j = 10, 25
        direct = scanner.run_config(i, j, sample_input)
        cached = scanner.cache_baseline_states(sample_input)
        from_cache = scanner.run_config_from_cache(i, j, cached)
        assert torch.allclose(direct.float(), from_cache.float(), atol=1e-4)

    def test_cached_config_differs_from_baseline(self, scanner, sample_input):
        cached = scanner.cache_baseline_states(sample_input)
        baseline_cls = cached[-1][:, 0]
        modified_cls = scanner.run_config_from_cache(10, 30, cached)
        assert not torch.allclose(baseline_cls.float(), modified_cls.float(), atol=1e-3)
