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
This gives L*(L+1)/2 configurations. For EVA-CLIP-18B (L=48), that is 1176 configs.

## Scoring (our adaptation)

For each (i, j) configuration:
1. Extract frozen embeddings (post-layernorm CLS token) from the modified model.
2. Score by nearest-centroid classification (cosine similarity, no training).
3. Compute Δ accuracy = config accuracy − baseline accuracy (no duplication).

NOTE: the original RYS work scored *generative behavior* of the full LLM
(math answers, EQ-Bench scores). We probe *embedding geometry* instead —
the natural interface of a contrastive vision encoder, but a deviation
worth keeping in mind: a null result here constrains representation
quality, not everything the model could do downstream.

## Heatmap Generation

Results are stored in a matrix M of shape (L, L+1) where M[i][j] = Δ accuracy
(valid cells satisfy 0 ≤ i < j ≤ L).
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

We apply RYS to the EVA-CLIP-18B vision encoder (17.5B parameters, 48 layers)
to investigate whether similar functional circuits exist in vision models, and
whether layer duplication improves or degrades visual representation quality.
The follow-up posts (RYS-II, Sapir-Whorf) motivate two additions: a
single-layer k-repeat scan, and a layer-anatomy analysis (style x content
grid) that independently locates the candidate "reasoning" phase.
