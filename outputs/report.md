# RYS Scan Report

- **Transformer layers:** 48
- **Configs evaluated:** 1176 / 1176
- **Probe:** nearest-centroid (cosine), zero training
- **Ranking basis:** z-scored Δaccuracy on the RANDOM control split, averaged across datasets (the borderline split is reported for contrast but is selection-sensitive)

## Probe Sets

| Dataset | Classes | Refs | Borderline n | Random n | BL acc (border) | BL acc (rand) | Max noise Δacc |
|---|---|---|---|---|---|---|---|
| counting | 10 | 150 | 100 | 100 | 0.500 | 0.390 | +0.000 |
| dtd | 47 | 235 | 100 | 100 | 0.500 | 0.750 | +0.006 |
| eurosat | 10 | 150 | 100 | 100 | 0.500 | 0.930 | +0.000 |
| fgvc_aircraft | 100 | 500 | 100 | 100 | 0.500 | 0.730 | +0.000 |
| imagenet | 100 | 500 | 100 | 100 | 0.500 | 0.960 | +0.002 |
| inside_outside | 2 | 100 | 100 | 100 | 0.500 | 0.900 | +0.009 |
| places365 | 100 | 500 | 100 | 100 | 0.500 | 0.600 | +0.000 |
| same_different | 2 | 100 | 100 | 100 | 0.500 | 0.820 | +0.007 |
| spatial_relation | 4 | 152 | 100 | 100 | 0.500 | 0.800 | +0.004 |
| stanford40 | 40 | 200 | 100 | 100 | 0.500 | 0.920 | +0.008 |
| symmetry | 2 | 100 | 100 | 100 | 1.000 | 1.000 | +0.000 |

*Max noise Δacc = largest mean Δaccuracy produced by pure iid Gaussian feature noise across tested magnitudes. This is a WEAK lower-bound null (duplication shifts features coherently, not iid); the matched null is the random-config control in stage-2 confirmation. Bootstrap CIs treat the datasets as fixed and are conditional on the reference sets.*

## Top 10 Configs (z-scored Δacc, random split)

| Rank | (i, j) | Block | z̄ | Δacc (rand) | 95% CI | Δacc (border) |
|---|---|---|---|---|---|---|
| 1 | (33, 46) | 13 | +0.827 | +0.0118 | [+0.001, +0.023] | +0.0845 |
| 2 | (25, 48) | 23 | +0.810 | +0.0100 | [-0.005, +0.025] | +0.0955 |
| 3 | (34, 46) | 12 | +0.803 | +0.0100 | [+0.000, +0.020] | +0.0800 |
| 4 | (30, 47) | 17 | +0.795 | +0.0091 | [-0.003, +0.021] | +0.0955 |
| 5 | (29, 47) | 18 | +0.792 | +0.0091 | [-0.005, +0.023] | +0.0773 |
| 6 | (33, 45) | 12 | +0.792 | +0.0091 | [-0.002, +0.021] | +0.0655 |
| 7 | (26, 47) | 21 | +0.788 | +0.0109 | [-0.005, +0.026] | +0.0900 |
| 8 | (34, 44) | 10 | +0.786 | +0.0091 | [-0.002, +0.020] | +0.0682 |
| 9 | (34, 42) | 8 | +0.783 | +0.0100 | [-0.002, +0.021] | +0.0882 |
| 10 | (24, 48) | 24 | +0.781 | +0.0073 | [-0.008, +0.022] | +0.0800 |

## Worst 5 Configs

| Rank | (i, j) | Block | z̄ | Δacc (rand) |
|---|---|---|---|---|
| 1 | (1, 3) | 2 | -5.278 | -0.4745 |
| 2 | (1, 2) | 1 | -4.284 | -0.4045 |
| 3 | (1, 4) | 3 | -4.021 | -0.4036 |
| 4 | (1, 46) | 45 | -2.950 | -0.3191 |
| 5 | (1, 5) | 4 | -2.901 | -0.3000 |

## Key Finding

**No config survived stage-2 confirmation** — on fresh images, no top config both excludes zero and beats the random-config null (-0.0441 ± 0.0586). Sweep-stage positives were selection noise.

### Best Config — Per-Dataset Δacc (random split)

| Dataset | Baseline | Config | Δ |
|---|---|---|---|
| counting | 0.390 | 0.450 | +0.060 |
| dtd | 0.750 | 0.730 | -0.020 |
| eurosat | 0.930 | 0.920 | -0.010 |
| fgvc_aircraft | 0.730 | 0.710 | -0.020 |
| imagenet | 0.960 | 0.960 | +0.000 |
| inside_outside | 0.900 | 0.930 | +0.030 |
| places365 | 0.600 | 0.610 | +0.010 |
| same_different | 0.820 | 0.830 | +0.010 |
| spatial_relation | 0.800 | 0.860 | +0.060 |
| stanford40 | 0.920 | 0.930 | +0.010 |
| symmetry | 1.000 | 1.000 | +0.000 |

## Summary Statistics (random split)

- Mean Δacc across configs: -0.0584
- Std: 0.0741
- Positive configs: 86 / 1176
- Max: +0.0118   Min: -0.4745

## Block Size Analysis (random split)

| Block size | Mean Δacc | Best config | Best Δacc | N |
|---|---|---|---|---|
| 1 | -0.0230 | (33, 34) | +0.0045 | 48 |
| 2 | -0.0283 | (46, 48) | +0.0018 | 47 |
| 3 | -0.0254 | (19, 22) | +0.0018 | 46 |
| 4 | -0.0229 | (34, 38) | +0.0064 | 45 |
| 5 | -0.0211 | (34, 39) | +0.0091 | 44 |
| 6 | -0.0163 | (34, 40) | +0.0064 | 43 |
| 7 | -0.0154 | (34, 41) | +0.0082 | 42 |
| 8 | -0.0165 | (34, 42) | +0.0100 | 41 |
| 9 | -0.0148 | (33, 42) | +0.0082 | 40 |
| 10 | -0.0154 | (34, 44) | +0.0091 | 39 |
| 11 | -0.0173 | (33, 44) | +0.0073 | 38 |
| 12 | -0.0174 | (34, 46) | +0.0100 | 37 |
| 13 | -0.0192 | (33, 46) | +0.0118 | 36 |
| 14 | -0.0213 | (33, 47) | +0.0064 | 35 |
| 15 | -0.0251 | (31, 46) | +0.0073 | 34 |
| 16 | -0.0287 | (31, 47) | +0.0064 | 33 |
| 17 | -0.0331 | (30, 47) | +0.0091 | 32 |
| 18 | -0.0380 | (29, 47) | +0.0091 | 31 |
| 19 | -0.0441 | (27, 46) | +0.0027 | 30 |
| 20 | -0.0467 | (26, 46) | +0.0064 | 29 |
| 21 | -0.0532 | (26, 47) | +0.0109 | 28 |
| 22 | -0.0602 | (24, 46) | +0.0091 | 27 |
| 23 | -0.0649 | (25, 48) | +0.0100 | 26 |
| 24 | -0.0701 | (24, 48) | +0.0073 | 25 |
| 25 | -0.0783 | (23, 48) | +0.0009 | 24 |
| 26 | -0.0828 | (22, 48) | -0.0109 | 23 |
| 27 | -0.0888 | (21, 48) | -0.0191 | 22 |
| 28 | -0.0920 | (6, 34) | -0.0173 | 21 |
| 29 | -0.0998 | (19, 48) | -0.0136 | 20 |
| 30 | -0.1072 | (4, 34) | -0.0236 | 19 |
| 31 | -0.1177 | (3, 34) | -0.0255 | 18 |
| 32 | -0.1320 | (3, 35) | -0.0391 | 17 |
| 33 | -0.1488 | (1, 34) | -0.0418 | 16 |
| 34 | -0.1607 | (0, 34) | -0.0445 | 15 |
| 35 | -0.1806 | (0, 35) | -0.0745 | 14 |
| 36 | -0.1994 | (0, 36) | -0.1073 | 13 |
| 37 | -0.2161 | (11, 48) | -0.1500 | 12 |
| 38 | -0.2282 | (10, 48) | -0.1791 | 11 |
| 39 | -0.2248 | (9, 48) | -0.1518 | 10 |
| 40 | -0.2340 | (8, 48) | -0.1764 | 9 |
| 41 | -0.2297 | (7, 48) | -0.1509 | 8 |
| 42 | -0.2365 | (6, 48) | -0.1364 | 7 |
| 43 | -0.2470 | (5, 48) | -0.1427 | 6 |
| 44 | -0.2411 | (4, 48) | -0.1482 | 5 |
| 45 | -0.2523 | (3, 48) | -0.1691 | 4 |
| 46 | -0.2333 | (2, 48) | -0.1827 | 3 |
| 47 | -0.1945 | (1, 48) | -0.1655 | 2 |
| 48 | -0.1455 | (0, 48) | -0.1455 | 1 |

## Layer Anatomy Cross-Reference

Anatomy analysis found no content-dominant region under either pooling — under the RYS hypothesis, duplication should mostly hurt this model.

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
