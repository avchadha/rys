# Methods — RYS Layer Duplication in a Large Vision Transformer

**Status: living document.** This records every experimental design decision
the experiment was ACTUALLY RUN with, and the rationale for each — both
decisions inherited from the original replication plan and the revisions
adopted after review. Superseded designs appear only in the Deviations
section when needed to interpret artifacts. Maintained as the source of
truth for the eventual paper; update it whenever the experimental procedure
changes. (`experiment_structure_decisions.txt` documents the pre-overhaul
design and is historical.)

---

## 1. Research questions and hypotheses

**RQ1 (transfer).** The RYS result (dnhkng, 2024–25) showed large LLMs
contain contiguous mid-stack layer blocks that can be executed twice at
inference time — no weight changes — with preserved or improved downstream
performance, while duplicating early/late layers is harmful. Does this
transfer to a frontier-scale pure Vision Transformer?

**RQ2 (mechanism).** The proposed mechanism (dnhkng "Sapir-Whorf" post) is
that duplicable layers are exactly those whose input and output live in the
same format-agnostic content space, so re-entry is a near-distribution
operation. Does a ViT have such a content-dominant band, and does its
location predict duplication tolerance?

**Registered prediction.** The anatomy experiment (§7) runs to completion
*before* the duplication sweep finishes, and its prediction is recorded
before results exist. Outcome on 2026-07-21 (pre-sweep): **no
content-dominant band exists in EVA-CLIP-18B under either pooling** (§7.4),
so the registered mechanistic prediction is: *duplication should be broadly
harmful at all depths, with no red mid-stack band.* All four
anatomy-x-heatmap outcome cells are interpretable (band+aligned tolerance =
mechanism transfers; band+no alignment = geometry story wrong;
no-band+broad harm = mechanism holds in negative form; no-band+tolerance =
mechanism story wrong).

---

## 2. Model

**BAAI/EVA-CLIP-18B, vision encoder only.** ~17.5B parameters (of 18B
total; the ~9B text tower is deleted at load), 48 transformer blocks,
hidden size 5120, 40 heads, patch size 14 at 224×224 → 257 tokens
(CLS at index 0), RMSNorm, learned absolute position embeddings (no RoPE),
fp16 inference.

*Rationale.* Largest publicly available pure ViT — maximizing scale keeps
the comparison honest against the 70B+ LLMs where RYS was demonstrated, and
48 layers give a rich 48·49/2 = 1176-point config space. A pure vision
encoder (vs a VLM) keeps the intervention attributable to visual
processing. CLIP-contrastive training gives an embedding space where cosine
similarity is the trained metric.

**Loading specifics** (verified against the repo's remote code,
`modeling_evaclip.py`):

- `AutoModel.from_pretrained(..., trust_remote_code=True, torch_dtype=fp16,
  low_cpu_mem_usage=True)` → `EvaCLIPModel`; we take `.vision_model`.
- Image processor from **`openai/clip-vit-large-patch14`**: the EVA repo
  ships no `preprocessor_config.json`; the model card itself uses the
  OpenAI CLIP processor. Transform: bicubic resize → center-crop 224 →
  normalize with the processor's mean/std.
- Encoder blocks have two *required* positional mask arguments
  (`forward(hidden_states, attention_mask, causal_attention_mask)`); the
  scanner calls `layer(h, None, None)` via a per-model `layer_fn` hook.
- **Feature space: post-layernorm CLS token.** `vision_model.post_layernorm`
  is applied inside the pooling function, uniformly for baseline and every
  config. *Rationale:* this is the model's trained output interface (the
  contrastive head consumes post-LN CLS); using one consistent space for
  baseline and configs makes deltas attributable solely to the layer
  intervention. (The final projection to the joint text-image space is not
  applied; post-LN CLS is the last vision-native representation.)

---

## 3. Intervention: layer duplication

**Config semantics.** For `(i, j)` with `0 ≤ i < j ≤ 48`: run blocks
`0..j-1` normally, feed the hidden state after block `j-1` back into block
`i`, re-run blocks `i..j-1`, then continue `j..47`. Weights unchanged.
Identical to the original RYS definition — a requirement for
comparability.

**Coverage: exhaustive.** All 1176 valid `(i, j)` pairs; no grid coarsening
or sampling. *Rationale:* any contiguous structure in the heatmap is then
real, not an interpolation artifact; narrow circuits (2–3 layers) cannot be
missed. Compute made tractable by the cached forward (below).

**Cached forward.** Per batch, one baseline pass caches all 49 hidden
states (embedding output + after each block). A config then starts from
`cached[j]` and runs only `(j−i) + (48−j)` blocks. Equivalence
`run_config_from_cache ≡ run_config` is covered by unit tests and asserted
against the real model in the pre-run sanity script.

**Single-layer k-repeat extension** (`--repeat-scan`). Every layer `i` also
run with total executions k ∈ {3, 4} (96 extra configs; k=2 is already the
`(i, i+1)` cell). *Rationale:* RYS-II found single layers repeated ~3×
can yield cheap gains. **Explicitly exploratory**: aggregate scores only,
no per-image persistence or CIs; excluded from headline claims and labeled
as such in figures.

---

## 4. Datasets (11)

All images pass through the model's native 224×224 transform. Class-capped
datasets use a deterministic random subset of classes
(`choose_class_subset`, seed 42, `--max-classes 100`).

### 4.1 Real-world (6) — each a distinct visual domain

| Dataset | Task | Classes used | Notes |
|---|---|---|---|
| ImageNet-1k | object recognition | 100 of 1000 (seeded cap) | HF `ILSVRC/imagenet-1k` (gated); cap for tractability while retaining a low-chance-level anchor task |
| DTD | texture | 47 | train+val merged as reference pool; test split for candidates |
| EuroSAT | satellite land use | 10 | 80/20 split, seeded generator; Zenodo fallback mirror |
| Places365 (small) | scene recognition | 100 of 365 (seeded cap) | train-standard/val |
| Stanford 40 | human actions | 40 | official splits; zip archives |
| FGVC Aircraft | fine-grained 3D structure | 100 | trainval/test |

*Rationale for the set:* each probes a different visual regime (objects,
textures without objects, overhead imagery, scene layout, pose+interaction,
subtle structural discrimination) so per-dataset heatmaps can reveal
operation-specific circuits — the vision analog of RYS's orthogonal
math-vs-EQ probes.

### 4.2 Synthetic visual-reasoning probes (5)

Procedurally generated at 448×448 (rendered from stored parameters;
deterministic; train/test generation seeds 42/123 → zero leakage), each
isolating one visual computation, each with explicit confound controls:

- **Counting** (10 classes, 1–10 dots). Radii drawn i.i.d. U{12..40}
  *before* placement, then positions rejection-sampled for strict
  non-overlap (≥6px gap), whole-layout restart on failure. *Controls:*
  non-overlap keeps the visible count equal to the label; radius-first
  sampling keeps the radius distribution count-independent; randomized dot
  sizes and background colors prevent a pure total-area shortcut. *Known
  residual:* total colored area still correlates with count — inherent to
  numerosity stimuli; noted for interpretation.
- **Same/Different** (2 classes). Two polygons; "same" = identical vertex
  set, translated and rotated ±0.6 rad; "different" = new random polygon
  with the same vertex count, **rescaled to match the left polygon's area
  exactly**. Colors always differ between the two shapes. *Controls:* color
  matching, vertex-count, and total-area shortcuts all removed; rotation
  forces abstract shape comparison rather than template matching.
- **Spatial relation** (4 classes: above/below/left/right). Red reference
  circle jittered ±70px around center; blue triangle at gap 80–160px with
  ±25px cross-axis jitter; clamping order preserves label validity.
  *Control:* reference jitter prevents absolute target position from fully
  determining the label. *Known residual:* jitter (±70) < max gap (160), so
  absolute position retains partial information; noted for interpretation.
- **Symmetry** (2 classes). Dot patterns; symmetric = exact mirror of the
  left half. Asymmetric images reuse the mirrored dots' exact (y, radius,
  color) multiset and resample only x. *Control:* color histograms, size
  distributions, dot counts, and vertical marginals are identical between
  halves for BOTH classes — only true spatial mirroring separates them.
- **Inside/Outside** (2 classes). Irregular 6–12-vertex star-shaped contour
  (radii 0.6–1.0·R, R∈[100,160]), drawn as outline; red dot placed with (a)
  boundary margin ≥ dot radius + line half-width + 8px so the label is
  visually unambiguous after downscaling, and (b) distance-from-center
  constrained to an annulus band [0.45R, 1.25R] where both classes can
  occur; contours regenerated when placement is infeasible; ray-casting
  ground truth. *Known residual:* the annulus weakens but does not
  eliminate the distance-from-center cue (inside dots concentrate below
  ~1.0R); documented, and this probe's results are interpreted with that
  caveat.

*Rationale for procedural stimuli:* zero download, exact reproducibility
from seeds, and complete control over shortcut cues — the confound fixes
above were adopted after adversarial review found the original versions
solvable via color-matching (symmetry), area (same/different), and merged
blobs (counting).

---

## 5. Probe methodology (scoring one config on one dataset)

**Classifier: nearest-centroid, cosine similarity, zero training.**
Class centroids are means of reference-image embeddings; test images are
assigned to the nearest centroid. *Rationale:* (a) matches the RYS ethos of
cheap, deterministic probes evaluable 1176+ times; (b) a trained probe
(e.g. logistic regression) can partially compensate for a degraded
embedding space and mask the effect being measured; (c) cosine is the
trained metric of a CLIP model. Centroids are recomputed *within each
config's feature space* — both refs and test images are re-embedded under
the modified forward pass, since comparing modified test features to
baseline centroids would conflate spaces.

**Reference sets.** Per class: `clamp(ceil(150 / n_classes), 5, 50)`
reference images (seed 42), i.e. ~5/class for 100-class datasets up to
50/class for binary probes. *Rationale:* centroid estimation noise scales
as 1/√n and is config-correlated (re-estimated per config); binary probes
with only 5 refs/class would make centroid jitter the dominant variance
source. The budget equalizes total centroid reliability across datasets at
negligible compute cost.

**Candidate pool.** 1000 random test-split images (seed 42, class-capped
consistently with the refs).

**Two test splits, both evaluated for every config:**

1. **Borderline split (n=100):** stratified 50 barely-misclassified + 50
   barely-correct candidates, ranked by |cosine margin| =
   |sim(true class) − max sim(other)| under the *baseline* model. Baseline
   accuracy on this split is ~50% by construction.
2. **Random control split (n=100):** uniform random from the remaining
   candidates, disjoint from the borderline split.

*Rationale.* The original plan selected the 50 *lowest-margin* images —
i.e. the most-confidently-wrong ones — which puts baseline accuracy near 0
on the test set and guarantees that ANY perturbation, including pure noise,
scores a positive delta (regression to the mean / adversarial-selection
bias). Stratifying around the decision boundary keeps selection's
sensitivity benefit (boundary images are where representation changes are
visible) while allowing deltas to move in both directions; the untouched
random split provides unbiased deltas and is the **sole ranking basis** for
all headline analyses. The borderline split is reported for contrast only.

**Noise-control null.** In Phase 1, per dataset: Gaussian noise of relative
magnitude α ∈ {0.005, 0.01, 0.02, 0.05, 0.1} (per-dimension σ =
α·mean‖f‖/√D) is added to *both* reference and test baseline features, 20
seeds each; the induced Δaccuracy distribution is recorded in
`probe_meta.json`. *Rationale:* an empirical floor showing what "no real
effect" produces on these exact probe sets. Documented as a **weak
lower-bound null** — duplication shifts features coherently, not i.i.d. —
with the random-config controls of §6 as the properly matched null.

**Metrics.** Top-1 accuracy and MRR (reciprocal rank of the true class,
ties resolved pessimistically against the true class). *Rationale:* MRR
captures sub-threshold representation improvements (rank 5 → rank 2) that
accuracy cannot see; both are reported everywhere.

**Per-image persistence.** For every (dataset, config, split): the
per-image correctness bitmap and true-class rank are persisted
(`*_percase.npz`). *Rationale:* enables paired bootstrap CIs and any
post-hoc reanalysis without re-running the GPU sweep; the original design
discarded per-image outcomes, making uncertainty quantification
impossible.

**Uncertainty.** Paired bootstrap: images resampled with replacement
within each dataset (config and baseline evaluated on the same resample, so
image difficulty cancels), per-dataset deltas averaged across datasets,
2000 resamples, 95% percentile intervals. *Scope caveat (stated in
report):* datasets are treated as fixed effects — conclusions generalize
to these tasks, not a task population — and CIs are conditional on the
reference sets.

**Aggregation across datasets.** Two aggregate heatmaps: (a) unweighted
mean of per-dataset Δ matrices (interpretable units), and (b) mean of
per-dataset **z-scored** Δ matrices, where each dataset's deltas are
standardized over its own 1176 valid cells. *Rationale:* raw deltas from a
binary probe (chance 50%, sampling SD ~5pp) and a 100-class dataset are
incommensurable; z-scoring prevents the noisiest probes from dominating the
mean. The z-scored random-split accuracy matrix is the canonical ranking
object.

---

## 6. Two-stage design (screen → confirm)

**Stage 1 — sweep.** All 1176 configs × 11 datasets × 2 splits, scored as
in §5. This stage *screens*; it cannot confirm: the top of a 1176-config
leaderboard evaluated on 100-image splits is inflated by selection
(expected max of ~1176 null cells with ~5pp SE is ≈ +15–20pp — winner's
curse).

**Stage 2 — confirmation** (`scripts/confirm_top.py`). The top 20 configs
by z-scored random-split aggregate, **plus 20 random control configs**
(seed 99), re-evaluated per dataset on:

- **fresh test images**: 500 per dataset, sampled (seed 777) from the test
  split *excluding the entire stage-1 candidate pool*;
- **resampled reference sets**: same per-class budget, seed 1337,
  disjoint from sweep references wherever class size allows. *Rationale:*
  sweep-stage winners were selected partly for lucky interaction with the
  sweep's specific few-shot centroids; reusing those refs would carry that
  selection component into confirmation.

**Positive-claim criterion (pre-specified):** a config counts as confirmed
only if its stage-2 paired-bootstrap 95% CI excludes zero **and** its
stage-2 delta exceeds the random-config control mean + 2 SD. The report
generator enforces this: the "Key Finding" section asserts an improvement
only from `confirm_results.json`; sweep-only results are labeled
**Unconfirmed** regardless of their CIs. *Rationale:* the random-config
controls are the correctly matched null (same refs, same fresh images, same
evaluation machinery, no selection), mirroring RYS-II's "small probes guide
search; large probes judge candidates."

---

## 7. Layer-anatomy experiment (mechanism probe)

### 7.1 Stimuli

`StyleContentGrid`: 8 shape classes (triangle, square, pentagon, hexagon,
5-point star, circle/40-gon, cross, arrow — defined as canonical vertex
geometry) × 8 rendering styles (solid-on-white, outline-only,
solid-on-black, stippled fill, striped fill, jittered multi-stroke sketch,
white-on-purple inverted, solid on cluttered noisy background) × 2 jittered
instances per cell = **128 images**, 448×448, deterministic from seed.
Per-instance jitter: rotation ±15°, translation ±12px, scale ×0.85–1.0;
style-internal randomness (stroke jitter ±4px, stipple/noise dot placement)
from the same per-cell seed.

*Rationale.* Direct translation of the Sapir-Whorf design (8 languages × 8
topics): style plays "language" (surface form), shape identity plays
"topic" (content). Jitter is calibrated to defeat pixel-level matching
between same-shape pairs (so similarity must come from abstraction) while
never making shape identity ambiguous (±15° cannot turn a pentagon into a
hexagon). Two instances per cell add a same-content-same-style ceiling
category and halve the variance of category means vs a single draw.

### 7.2 Measurement

For every layer (embedding output + after each of 48 blocks): pool each
image's tokens (**CLS primary**; masked mean over patch tokens secondary),
compute all C(128,2) pairwise cosine similarities, **center per layer**
(subtract the layer's mean pairwise similarity), and average within pair
categories: same-content/different-style, same-style/different-content,
different-both, same-both (ceiling). The **content-dominant band** is the
set of layers where the same-content curve exceeds the same-style curve.
Auxiliary visualization: per-layer 2-component PCA scatter colored by
content, markered by style.

*Rationale for CLS-primary:* the style axis manipulates mostly background
(~75% of canvas area), so mean-patch pooling measures background statistics
at every depth and can fake style-organization regardless of the model's
internal structure; CLS is the model's own learned global summary and is
also the exact vector the duplication probes use, so its geometry is the
one the mechanistic claim is about. Mean-patch curves are retained as a
secondary view with this caveat attached. Per-layer centering removes
layer-wise offsets in similarity scale (as in the source post).

### 7.3 Registered-prediction protocol

The anatomy runs as stage 1 of the pipeline, completing before any
duplication result exists; `anatomy.json` is the timestamped artifact. The
report cross-references the content-dominant band against the sweep
heatmaps (§1, RQ2).

### 7.4 Anatomy outcome (recorded 2026-07-21, pre-sweep)

**No content-dominant band under either pooling.** CLS: the
content-minus-style gap is ≤ 0 at every layer (0.0 at the embedding, where
CLS is input-independent; −0.155 at the final layer). Mean-patch: −0.69 at
layer 0 (background-driven), rising monotonically to ≈ −0.03 by layer 45,
never crossing zero. **Registered prediction: duplication broadly harmful
at all depths.** Interpretive caveat recorded alongside: for an image-text
contrastive model, rendering style and background color are legitimately
caption-relevant *content*, so style-dominance in CLIP space is not
automatically a failure to abstract; the sweep adjudicates (§1).

---

## 8. Execution, reproducibility, and engineering controls

- **Hardware/software.** Single H100 80GB (Lambda), fp16, batch 64,
  `torch.inference_mode()`. Pinned: `torch 2.13+cu130`,
  `transformers 4.49 (<4.50: EVA remote code is early-2024 vintage)`,
  `datasets 2.21 (<3: script-based ImageNet loader removed in 3.x)`,
  `numpy <2`, `accelerate` (required by `low_cpu_mem_usage`).
- **Determinism.** All sampling seeded: reference/candidate selection 42,
  random-split 4242, confirmation test images 777, confirmation refs 1337,
  control configs 99, synthetic train/test generation 42/123, EuroSAT split
  generator 42, class caps 42. Model in eval mode, no dropout. Residual
  fp16 GPU nondeterminism is sub-quantization for 100-image accuracy.
- **Probe-set persistence.** Phase 1 writes all selected indices
  (`probe_sets.json`) and baseline features/scores
  (`*_baseline_features.npz`) before any config is scored; resume restores
  the *identical* probe sets rather than re-deriving them (near-tie margins
  could otherwise reorder under fp16 across runs, silently switching test
  sets mid-scan).
- **Resume correctness.** Completion is judged from NaN cells across all
  four matrices (both splits × both metrics); per-image arrays are written
  *before* matrices so the completion signal is the last write; a simulated
  mid-scan crash was verified to repair bit-identically under `--resume`.
- **Dataset-loading policy.** Failures raise by default; the orchestrated
  sweep passes `--allow-missing-datasets` so one flaky mirror cannot abort
  a multi-day run (missing datasets join via `--resume` later). Stanford40
  uses the current .zip archives; EuroSAT falls back to the official Zenodo
  record when the primary mirror is down.
- **Pipeline order.** sanity check (exercises the exact embed→layers→pool
  path) → anatomy → single-dataset smoke scan → full sweep
  (cheapest-dataset-first for crash resilience) → k-repeats → stage-2
  confirmation → figures + report. Everything idempotent/resumable
  (`scripts/run_h100.sh`).
- **Verification.** 22-test CPU suite covering scanner math (cache ≡
  direct, repeat semantics, validation), probe scoring, borderline
  stratification, every synthetic confound control (non-overlap,
  area-matching, half-marginal identity, boundary margins, label
  validity), anatomy pair-category combinatorics and
  planted-structure recovery, plus an end-to-end miniature pipeline
  (scan → resume → figures → report → confirmation) with a dummy
  transformer. GPU integration tests assert the real model's layer count,
  shapes, and cache/direct equivalence before the sweep.
- **Provenance.** Code at `github.com/avchadha/rys` (master). Sweep run
  started 2026-07-21 on `DATASETS=all` (11 datasets; ImageNet access
  granted before the sweep stage loaded datasets, so ImageNet-100 is
  included in the main sweep rather than appended).

## 9. Presentation conventions

Upper-triangular heatmaps identical in convention to the RYS posts (y = i,
x = j, RdBu_r diverging colormap centered at 0, red = improvement, top-5
annotated), per dataset × split × metric, plus raw and z-scored aggregates;
Δ-vs-block-size Pareto scatter with frontier (RYS-II convention);
single-layer k-repeat curves labeled "exploratory — no CIs"; anatomy
three-curve plot (+ ceiling curve) with the content-dominant region shaded,
and per-layer PCA panels.

## 10. Known limitations (to state in the paper)

1. Embedding-geometry probes, not behavioral ones: the original RYS scored
   generative task performance of the full model; we score representation
   quality at the encoder's output interface. A null here constrains
   representation quality, not every downstream use.
2. Datasets treated as fixed effects; CIs conditional on reference sets.
3. Synthetic-probe residual cues: inside/outside distance-from-center
   (weakened, not eliminated), spatial-relation absolute position
   (partial), counting total-area (inherent).
4. Anatomy stimulus caveat: style-vs-content is operationalized with
   procedural stimuli where, for a contrastive model, style is arguably
   content (§7.4).
5. Single model; k-repeat scan exploratory; one H100 fp16 run (no seed
   replication of the model forward itself — deterministic by design).
