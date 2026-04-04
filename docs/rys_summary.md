# RYS (Repeat Yourself) — Layer Duplication for Transformers

## Technique

RYS duplicates a contiguous block of transformer layers at inference time.
Given a model with L layers (indexed 0 to L-1), a configuration (i, j) means:

1. Run layers 0 through j-1 normally.
2. Take the hidden state after layer j-1.
3. Feed it back into layer i and re-run layers i through j-1.
4. Continue with layers j through L-1.

The duplicated block spans layers [i, j) — that is, layers i, i+1, …, j-1 are
executed twice. No weights are changed; the same parameters are reused.

## Configuration Space

For a model with L layers, valid configs satisfy 0 ≤ i < j ≤ L.
This gives L*(L+1)/2 configurations. For InternViT-6B (L=45), that is 1035 configs.

## Scoring with Linear Probes

For each (i, j) configuration:
1. Extract frozen embeddings (CLS token) from the modified model on a dataset.
2. Train a linear classifier (logistic regression) on these embeddings.
3. Record top-1 classification accuracy.
4. Compute Δ accuracy = config accuracy − baseline accuracy (no duplication).

## Heatmap Generation

Results are stored in a matrix M of shape (L, L) where M[i][j] = Δ accuracy.
The matrix is visualized as a 2D heatmap:
- X-axis: j (end of duplicated block)
- Y-axis: i (start of duplicated block)
- Red: improvement over baseline
- Blue: degradation vs baseline

## Key Finding from the Original RYS Work (on LLMs)

- Middle-layer circuits of 7+ contiguous layers behave as indivisible functional
  units in large language models.
- Duplicating these circuits can improve reasoning performance.
- Early and late layers are more specialized and less tolerant of duplication.

## Our Experiment

We apply RYS to InternViT-6B (a 6-billion parameter Vision Transformer) to
investigate whether similar functional circuits exist in vision models, and whether
layer duplication improves or degrades visual representation quality.
