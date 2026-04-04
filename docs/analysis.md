# RYS EVA-CLIP-18B — Analysis

## Reproduction Steps

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Verify model loading:
   ```bash
   python scripts/test_model_loading.py
   ```

3. Run the full scan across all benchmark datasets:
   ```bash
   python scripts/run_scan.py --data-dir /path/to/data_root
   ```
   For a quick test on a subset:
   ```bash
   python scripts/run_scan.py --data-dir /path/to/data --datasets dtd eurosat --n-hard 20
   ```

4. Generate heatmaps (per-dataset + aggregated, accuracy + MRR):
   ```bash
   python visualization/heatmap.py
   ```

5. Generate the final report:
   ```bash
   python scripts/report.py
   ```

6. Run unit tests:
   ```bash
   pytest tests/ -v
   ```

## Model Details

- **Model:** `BAAI/EVA-CLIP-18B` (vision encoder only)
- **Parameters:** ~17.5B (vision encoder); text encoder discarded at load time
- **Transformer layers:** 48
- **Hidden size:** 5120
- **Attention heads:** 40
- **Input resolution:** 224x224 (patch size 14, sequence length 257)
- **Total RYS configs:** 1176
- **Normalization:** RMSNorm (LLaMA-style)
- **dtype:** float16

## Probe Methodology

Following the original RYS blog post, we use lightweight probes that require
no classifier training:

1. **Reference set:** 5 images per class per dataset (from training split),
   used to compute class centroids in embedding space.
2. **Hard test set:** 50 images per dataset selected by lowest cosine-similarity
   margin between correct and best-incorrect class on the baseline model.
3. **Scoring:** Nearest-centroid classification via cosine similarity.
   Metrics: top-1 accuracy and mean reciprocal rank (MRR).
4. **Per config:** Re-extract embeddings with layer duplication, recompute
   centroids, re-score. No training at any point.

Total probe images: ~8600 reference + 550 hard test across 11 datasets.

## Benchmark Datasets

### Real-world datasets

| Dataset | Task | Classes | Download |
|---------|------|---------|----------|
| ImageNet-1K | Object classification | 1000 | Manual (image-net.org) |
| DTD | Texture recognition | 47 | Auto |
| EuroSAT | Satellite land use | 10 | Auto |
| Places365 | Scene recognition | 365 | Auto (large: ~30GB) |
| Stanford 40 | Action recognition | 40 | Auto (~1GB) |
| FGVC Aircraft | Viewpoint / 3D structure | 100 | Auto |

### Synthetic reasoning probes (no download needed)

Each probe isolates a specific visual cognitive operation:

| Probe | Operation Tested | Classes | Confound Controls |
|-------|-----------------|---------|-------------------|
| Counting | Numerosity perception | 10 (1-10 dots) | Randomized dot sizes prevent area shortcut |
| Same/Different | Relational identity | 2 | Matched vertex count; different colors; slight rotation on "same" |
| Spatial Relation | Spatial reasoning | 4 (above/below/left/right) | Reference position jittered; cross-axis noise |
| Symmetry | Global structure perception | 2 | Asymmetric has same dot count per half |
| Inside/Outside | Topological reasoning | 2 | Ray-casting ground truth; irregular contours |

### Manual download instructions

**ImageNet-1K:** Register at https://image-net.org, download ILSVRC2012 train and val
sets, extract into `{data-dir}/imagenet/` with `train/` and `val/` subdirectories.

## Results

*See `outputs/report.md` after running the scan.*

## Comparison with LLM Findings

The original RYS work on LLMs found:
- Middle-layer circuits of 7+ layers are indivisible functional units
- Duplicating these circuits improves reasoning performance
- Early and late layers are specialized and duplication-intolerant

For EVA-CLIP-18B (the largest publicly available pure ViT, trained with CLIP):
- Key differences: no autoregressive objective, patch-level attention, CLS token aggregation
- With 48 layers and 17.5B params, there is ample room for circuit structure
- Uses RMSNorm and LLaMA-style architecture conventions
- TBD: whether similar circuit structure exists in vision models at this scale
