"""RYS (Repeat Yourself) layer duplication scanner.

Model-agnostic: works with any transformer by providing three callables
that describe how to access the model's layers, embedding function, and
pooling strategy.  Defaults to InternViT-6B conventions.

Usage with InternViT-6B (defaults):
    scanner = RYSScanner(model, device="cuda")

Usage with a different model (e.g. DINOv2-large):
    scanner = RYSScanner(
        model, device="cuda",
        get_layers=lambda m: list(m.blocks),
        embed_fn=lambda m, x: m.prepare_tokens(x),
        pool_fn=lambda h: h[:, 0],
    )
"""

from collections.abc import Callable

import torch
import torch.nn as nn


class RYSScanner:
    """Applies the RYS layer-duplication technique to a transformer model.

    For config (i, j), transformer blocks [i, j) are executed twice:
    normal forward through all blocks, then the hidden state after block j-1
    is fed back into block i and blocks i..j-1 run again.

    Args:
        model: A transformer model.
        device: Device to run on.
        get_layers: Callable(model) -> list of transformer blocks.
                    Default: list(model.encoder.layers)
        embed_fn:   Callable(model, inputs) -> initial hidden states tensor.
                    Default: model.embeddings(inputs)
        pool_fn:    Callable(hidden_states) -> pooled embedding tensor.
                    Default: hidden_states[:, 0]  (CLS token)
        layer_fn:   Callable(layer, hidden_states) -> layer output. Use this
                    when blocks need extra positional args, e.g. EVA-CLIP's
                    EvaCLIPEncoderLayer.forward(hs, attention_mask,
                    causal_attention_mask) requires
                    lambda l, h: l(h, None, None).
                    Default: layer(hidden_states)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        get_layers: Callable | None = None,
        embed_fn: Callable | None = None,
        pool_fn: Callable | None = None,
        layer_fn: Callable | None = None,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self._dtype = getattr(model, "dtype", next(model.parameters()).dtype)

        # Model-specific accessors (defaults work for InternViT-6B and EVA-CLIP-18B)
        self._embed_fn = embed_fn or (lambda m, x: m.embeddings(x))
        self._pool_fn = pool_fn or (lambda h: h[:, 0])
        self._layer_fn = layer_fn or (lambda layer, h: layer(h))

        _get_layers = get_layers or (lambda m: list(m.encoder.layers))
        self._layers = _get_layers(self.model)
        self.num_layers = len(self._layers)

    def _run_layer(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._unwrap(self._layer_fn(self._layers[layer_idx], hidden_states))

    @staticmethod
    def _unwrap(output):
        """Unwrap layer output: some layers return (tensor,) tuples, others return tensor."""
        return output[0] if isinstance(output, (tuple, list)) else output

    @torch.no_grad()
    def get_baseline_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get pooled embeddings from unmodified model."""
        pixel_values = pixel_values.to(device=self.device, dtype=self._dtype)
        hidden_states = self._embed_fn(self.model, pixel_values)
        for layer_idx in range(self.num_layers):
            hidden_states = self._run_layer(layer_idx, hidden_states)
        return self._pool_fn(hidden_states)

    @torch.no_grad()
    def cache_baseline_states(
        self, pixel_values: torch.Tensor
    ) -> list[torch.Tensor]:
        """Run the baseline forward pass and cache hidden states after each block.

        Returns a list of length num_layers + 1:
          cached[0] = output of embeddings (input to block 0)
          cached[k] = hidden state after block k-1 (input to block k)
          cached[num_layers] = final hidden state
        """
        pixel_values = pixel_values.to(device=self.device, dtype=self._dtype)
        hidden_states = self._embed_fn(self.model, pixel_values)

        cached = [hidden_states]
        for layer_idx in range(self.num_layers):
            hidden_states = self._run_layer(layer_idx, hidden_states)
            cached.append(hidden_states)

        return cached

    @torch.no_grad()
    def run_config_from_cache(
        self,
        i: int,
        j: int,
        cached_states: list[torch.Tensor],
        repeats: int = 2,
    ) -> torch.Tensor:
        """Run config (i, j) using pre-cached baseline hidden states.

        Starts from the cached state after block j-1, re-runs blocks [i, j)
        `repeats - 1` more times (the cache already contains the first pass),
        then continues blocks [j, end). Returns pooled embedding.
        """
        if not self.validate_config(i, j):
            raise ValueError(
                f"Invalid config ({i}, {j}): need 0 <= i < j <= {self.num_layers}"
            )
        if repeats < 2:
            raise ValueError(f"repeats must be >= 2, got {repeats}")

        # Start from cached state after block j-1
        hidden_states = cached_states[j]

        # Duplicate: re-run blocks i through j-1, (repeats - 1) extra passes
        for _ in range(repeats - 1):
            for layer_idx in range(i, j):
                hidden_states = self._run_layer(layer_idx, hidden_states)

        # Continue: blocks j through end
        for layer_idx in range(j, self.num_layers):
            hidden_states = self._run_layer(layer_idx, hidden_states)

        return self._pool_fn(hidden_states)

    @torch.no_grad()
    def run_config(
        self, i: int, j: int, pixel_values: torch.Tensor, repeats: int = 2
    ) -> torch.Tensor:
        """Run model with blocks [i, j) executed `repeats` times total.

        Args:
            i: Start block index (inclusive).
            j: End block index (exclusive). Must satisfy 0 <= i < j <= num_layers.
            pixel_values: Input tensor (e.g. images).
            repeats: Total executions of the block (2 = classic RYS duplication).

        Returns:
            Pooled embeddings, shape (B, hidden_dim).
        """
        if not (0 <= i < j <= self.num_layers):
            raise ValueError(
                f"Invalid config ({i}, {j}): need 0 <= i < j <= {self.num_layers}"
            )
        if repeats < 2:
            raise ValueError(f"repeats must be >= 2, got {repeats}")

        pixel_values = pixel_values.to(device=self.device, dtype=self._dtype)
        hidden_states = self._embed_fn(self.model, pixel_values)

        # First pass: blocks 0 through j-1
        for layer_idx in range(j):
            hidden_states = self._run_layer(layer_idx, hidden_states)

        # Duplicate: re-run blocks i through j-1, (repeats - 1) extra passes
        for _ in range(repeats - 1):
            for layer_idx in range(i, j):
                hidden_states = self._run_layer(layer_idx, hidden_states)

        # Continue: blocks j through end
        for layer_idx in range(j, self.num_layers):
            hidden_states = self._run_layer(layer_idx, hidden_states)

        return self._pool_fn(hidden_states)

    def validate_config(self, i: int, j: int) -> bool:
        """Check if (i, j) is a valid duplication config."""
        return 0 <= i < j <= self.num_layers

    def all_configs(self):
        """Yield all valid (i, j) pairs."""
        for i in range(self.num_layers):
            for j in range(i + 1, self.num_layers + 1):
                yield (i, j)

    def single_layer_repeat_configs(self, k_values=(3, 4)):
        """Yield (i, i+1, k) configs: layer i executed k times total.

        k=2 is already covered by the standard (i, i+1) scan; this adds the
        higher repeat counts explored in RYS-II, where e.g. one layer
        repeated 3x gave cheap gains.
        """
        for k in k_values:
            for i in range(self.num_layers):
                yield (i, i + 1, k)

    def num_configs(self) -> int:
        """Total number of valid (i, j) configurations."""
        n = self.num_layers
        return n * (n + 1) // 2
