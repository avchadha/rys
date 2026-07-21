# RYS for Vision Transformers

Does the [RYS layer-duplication result](https://dnhkng.github.io/posts/rys/)
— that LLMs contain contiguous mid-stack "reasoning circuits" which can be
executed twice at inference time for free accuracy — transfer to large
Vision Transformers?

Target model: **BAAI/EVA-CLIP-18B** vision encoder (17.5B params, 48 layers
→ 1176 duplication configs), the largest public pure ViT.

## Method

For config `(i, j)`, blocks `[i, j)` run twice: `0..j-1, i..j-1, j..end`.
No weights change. Each config is scored by nearest-centroid classification
(cosine similarity, zero training) over 10 datasets — 5 real-world domains
(DTD textures, EuroSAT satellite, Places365 scenes, Stanford40 actions,
FGVC aircraft) and 5 procedurally generated visual-reasoning probes
(counting, same/different, spatial relations, symmetry, inside/outside),
each with documented confound controls. ImageNet is supported but opt-in
(gated + 155 GB).

### Bias controls (what makes the deltas trustworthy)

- **Two test splits per dataset**: a *borderline* split (stratified
  barely-right/barely-wrong, baseline ≈ 50%) and a *random control* split.
  Rankings use the random split — selecting only hardest images guarantees
  fake positive deltas via regression to the mean.
- **Noise control**: Phase 1 measures Δaccuracy under pure Gaussian feature
  noise; a config that doesn't beat the noise null shows nothing.
- **Per-image outcomes persisted** → paired-bootstrap 95% CIs in the report.
- **Stage 2 confirmation** (`scripts/confirm_top.py`): top configs + random
  control configs re-scored on fresh, never-touched images — the winner's
  curse from ranking 1176 configs on ~100 images dies here.
- **z-scored aggregation** across datasets so binary probes and 100-class
  datasets are commensurable.

### Extensions beyond the original plan

- **Layer anatomy** (`scripts/run_anatomy.py`): ViT analog of the
  [Sapir-Whorf post](https://dnhkng.github.io/posts/sapir-whorf/) — an
  8-shape x 8-rendering-style grid replaces 8-topic x 8-language sentences;
  per-layer centered cosine similarity locates the content-dominant
  ("reasoning") phase, predicting where duplication should be tolerated.
- **Single-layer k-repeats** (`--repeat-scan`): layers repeated 3-4x, as in
  [RYS-II](https://dnhkng.github.io/posts/rys-ii/).
- **Pareto view**: Δaccuracy vs duplicated-block compute overhead.

## Running

```bash
pip install -r requirements.txt

# everything, resumable (H100 recommended; ~12-16 h for the default suite):
bash scripts/run_h100.sh 2>&1 | tee run.log

# or piecewise:
python scripts/test_model_loading.py                # sanity + throughput
python scripts/run_anatomy.py                       # ~5 min
python scripts/run_scan.py --data-dir ~/rys_data --repeat-scan --resume
python scripts/confirm_top.py --data-dir ~/rys_data
python visualization/heatmap.py && python scripts/report.py
```

Tests: `pytest tests/test_cpu.py tests/test_pipeline_cpu.py` runs the whole
logic layer + an end-to-end miniature pipeline on CPU (no model download);
`pytest tests/test_layer_loop.py` needs the GPU box.

## Layout

```
scanner/        RYSScanner (cached layer-loop forward) + anatomy analysis
benchmarks/     dataset suite, synthetic probes, style-content grid
probes/         nearest-centroid scoring, borderline selection
scripts/        run_scan, confirm_top, run_anatomy, report, model_registry
visualization/  heatmaps, Pareto, repeat curves
results/        matrices, per-image outcomes, probe sets (generated)
outputs/        PNGs + report.md (generated)
```
