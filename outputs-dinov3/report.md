# RYS Scan Report

- **Transformer layers:** 40
- **Configs evaluated:** 820 / 820
- **Probe:** nearest-centroid (cosine), zero training
- **Ranking basis:** z-scored Δaccuracy on the RANDOM control split, averaged across datasets (the borderline split is reported for contrast but is selection-sensitive)

## Probe Sets

| Dataset | Classes | Refs | Borderline n | Random n | BL acc (border) | BL acc (rand) | Max noise Δacc |
|---|---|---|---|---|---|---|---|
| counting | 10 | 150 | 100 | 100 | 0.500 | 0.420 | +0.000 |
| dtd | 47 | 235 | 100 | 100 | 0.500 | 0.750 | +0.000 |
| eurosat | 10 | 150 | 100 | 100 | 0.500 | 0.950 | +0.011 |
| fgvc_aircraft | 100 | 500 | 100 | 100 | 0.500 | 0.450 | +0.009 |
| imagenet | 100 | 500 | 100 | 100 | 0.500 | 0.990 | -0.000 |
| inside_outside | 2 | 100 | 100 | 100 | 0.940 | 1.000 | +0.000 |
| places365 | 100 | 500 | 100 | 100 | 0.500 | 0.610 | +0.003 |
| same_different | 2 | 100 | 100 | 100 | 0.500 | 0.930 | +0.025 |
| spatial_relation | 4 | 152 | 100 | 100 | 0.500 | 0.990 | -0.000 |
| stanford40 | 40 | 200 | 100 | 100 | 0.500 | 0.960 | +0.018 |
| symmetry | 2 | 100 | 100 | 100 | 0.910 | 1.000 | +0.000 |

*Max noise Δacc = largest mean Δaccuracy produced by pure iid Gaussian feature noise across tested magnitudes. This is a WEAK lower-bound null (duplication shifts features coherently, not iid); the matched null is the random-config control in stage-2 confirmation. Bootstrap CIs treat the datasets as fixed and are conditional on the reference sets.*

## Top 10 Configs (z-scored Δacc, random split)

| Rank | (i, j) | Block | z̄ | Δacc (rand) | 95% CI | Δacc (border) |
|---|---|---|---|---|---|---|
| 1 | (1, 32) | 31 | +0.301 | +0.0064 | [-0.003, +0.015] | +0.0455 |
| 2 | (2, 32) | 30 | +0.301 | +0.0064 | [-0.003, +0.015] | +0.0409 |
| 3 | (2, 31) | 29 | +0.295 | +0.0055 | [-0.004, +0.015] | +0.0345 |
| 4 | (3, 32) | 29 | +0.270 | +0.0045 | [-0.005, +0.014] | +0.0355 |
| 5 | (0, 31) | 31 | +0.268 | +0.0036 | [-0.006, +0.015] | +0.0473 |
| 6 | (13, 31) | 18 | +0.259 | +0.0055 | [-0.001, +0.012] | +0.0345 |
| 7 | (8, 32) | 24 | +0.256 | +0.0045 | [-0.004, +0.014] | +0.0245 |
| 8 | (3, 31) | 28 | +0.255 | +0.0045 | [-0.004, +0.014] | +0.0273 |
| 9 | (9, 32) | 23 | +0.252 | +0.0045 | [-0.004, +0.014] | +0.0345 |
| 10 | (6, 32) | 26 | +0.249 | +0.0036 | [-0.005, +0.013] | +0.0264 |

## Worst 5 Configs

| Rank | (i, j) | Block | z̄ | Δacc (rand) |
|---|---|---|---|---|
| 1 | (0, 2) | 2 | -11.200 | -0.6500 |
| 2 | (0, 3) | 3 | -10.923 | -0.6391 |
| 3 | (0, 5) | 5 | -10.914 | -0.6382 |
| 4 | (0, 4) | 4 | -10.914 | -0.6382 |
| 5 | (0, 1) | 1 | -10.914 | -0.6382 |

## Key Finding

**No config survived stage-2 confirmation** — on fresh images, no top config both excludes zero and beats the random-config null (-0.0005 ± 0.0047). Sweep-stage positives were selection noise.

### Best Config — Per-Dataset Δacc (random split)

| Dataset | Baseline | Config | Δ |
|---|---|---|---|
| counting | 0.420 | 0.480 | +0.060 |
| dtd | 0.750 | 0.750 | +0.000 |
| eurosat | 0.950 | 0.950 | +0.000 |
| fgvc_aircraft | 0.450 | 0.460 | +0.010 |
| imagenet | 0.990 | 0.990 | +0.000 |
| inside_outside | 1.000 | 1.000 | +0.000 |
| places365 | 0.610 | 0.600 | -0.010 |
| same_different | 0.930 | 0.950 | +0.020 |
| spatial_relation | 0.990 | 0.990 | +0.000 |
| stanford40 | 0.960 | 0.950 | -0.010 |
| symmetry | 1.000 | 1.000 | +0.000 |

## Summary Statistics (random split)

- Mean Δacc across configs: -0.0067
- Std: 0.0559
- Positive configs: 228 / 820
- Max: +0.0064   Min: -0.6500

## Block Size Analysis (random split)

| Block size | Mean Δacc | Best config | Best Δacc | N |
|---|---|---|---|---|
| 1 | -0.0159 | (23, 24) | +0.0018 | 40 |
| 2 | -0.0165 | (21, 23) | +0.0018 | 39 |
| 3 | -0.0167 | (26, 29) | +0.0036 | 38 |
| 4 | -0.0170 | (22, 26) | +0.0036 | 37 |
| 5 | -0.0174 | (26, 31) | +0.0055 | 36 |
| 6 | -0.0178 | (21, 27) | +0.0055 | 35 |
| 7 | -0.0093 | (20, 27) | +0.0055 | 34 |
| 8 | -0.0032 | (20, 28) | +0.0045 | 33 |
| 9 | -0.0005 | (23, 32) | +0.0036 | 32 |
| 10 | -0.0003 | (20, 30) | +0.0045 | 31 |
| 11 | -0.0005 | (19, 30) | +0.0045 | 30 |
| 12 | -0.0006 | (18, 30) | +0.0045 | 29 |
| 13 | -0.0007 | (17, 30) | +0.0045 | 28 |
| 14 | -0.0010 | (16, 30) | +0.0036 | 27 |
| 15 | -0.0009 | (16, 31) | +0.0045 | 26 |
| 16 | -0.0009 | (15, 31) | +0.0045 | 25 |
| 17 | -0.0012 | (14, 31) | +0.0036 | 24 |
| 18 | -0.0013 | (13, 31) | +0.0055 | 23 |
| 19 | -0.0016 | (12, 31) | +0.0036 | 22 |
| 20 | -0.0018 | (12, 32) | +0.0027 | 21 |
| 21 | -0.0018 | (11, 32) | +0.0045 | 20 |
| 22 | -0.0024 | (10, 32) | +0.0036 | 19 |
| 23 | -0.0030 | (9, 32) | +0.0045 | 18 |
| 24 | -0.0028 | (8, 32) | +0.0045 | 17 |
| 25 | -0.0032 | (6, 31) | +0.0027 | 16 |
| 26 | -0.0039 | (6, 32) | +0.0036 | 15 |
| 27 | -0.0047 | (4, 31) | +0.0036 | 14 |
| 28 | -0.0046 | (3, 31) | +0.0045 | 13 |
| 29 | -0.0048 | (2, 31) | +0.0055 | 12 |
| 30 | -0.0045 | (2, 32) | +0.0064 | 11 |
| 31 | -0.0047 | (1, 32) | +0.0064 | 10 |
| 32 | -0.0071 | (0, 32) | +0.0018 | 9 |
| 33 | -0.0085 | (0, 33) | -0.0018 | 8 |
| 34 | -0.0103 | (0, 34) | -0.0064 | 7 |
| 35 | -0.0098 | (1, 36) | -0.0073 | 6 |
| 36 | -0.0095 | (0, 36) | -0.0045 | 5 |
| 37 | -0.0109 | (0, 37) | -0.0082 | 4 |
| 38 | -0.0112 | (2, 40) | -0.0073 | 3 |
| 39 | -0.0100 | (1, 40) | -0.0073 | 2 |
| 40 | -0.0073 | (0, 40) | -0.0073 | 1 |

## Layer Anatomy Cross-Reference

Content-dominant (candidate 'reasoning') layers — CLS pooling: layers 35–40; mean-patch pooling: layers 28–40. The Sapir-Whorf/RYS hypothesis predicts duplication tolerance inside this range and damage outside it — compare with the heatmaps. (CLS is the primary curve; mean-patch is background-sensitive.)

## Heatmaps

### counting
![counting rand](heatmap_counting_rand_accuracy.png)

### dtd
![dtd rand](heatmap_dtd_rand_accuracy.png)

### eurosat
![eurosat rand](heatmap_eurosat_rand_accuracy.png)

### fgvc_aircraft
![fgvc_aircraft rand](heatmap_fgvc_aircraft_rand_accuracy.png)

### imagenet
![imagenet rand](heatmap_imagenet_rand_accuracy.png)

### inside_outside
![inside_outside rand](heatmap_inside_outside_rand_accuracy.png)

### places365
![places365 rand](heatmap_places365_rand_accuracy.png)

### same_different
![same_different rand](heatmap_same_different_rand_accuracy.png)

### spatial_relation
![spatial_relation rand](heatmap_spatial_relation_rand_accuracy.png)

### stanford40
![stanford40 rand](heatmap_stanford40_rand_accuracy.png)

### symmetry
![symmetry rand](heatmap_symmetry_rand_accuracy.png)

### Aggregated (rand, z-scored)
![aggregated](heatmap_aggregated_rand_accuracy_zscore.png)

### Pareto
![pareto](pareto_rand_accuracy.png)
