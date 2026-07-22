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

## 0. Provenance: source material and what was drawn from each

The experiment adapts the RYS blog series (D. Hankins / dnhkng) to vision:

- **RYS I** (https://dnhkng.github.io/posts/rys/): the core intervention —
  brute-force scan of all contiguous-block duplication configs `(i, j)` on
  Qwen2-72B, scored with tiny deterministic probes (16 math items, 16
  EQ-Bench items; minimal output tokens, no LLM-as-judge), best config
  (45, 52) improving 5/6 leaderboard benchmarks. Adopted: exact config
  semantics; exhaustive enumeration; the small-deterministic-probe
  philosophy (our nearest-centroid probes are its embedding-space analog);
  the upper-triangular Δ heatmap presentation; the finding to test —
  "reasoning circuits" of ≥7 contiguous mid-stack layers, with single-layer
  duplication almost always harmful.
- **RYS II** (https://dnhkng.github.io/posts/rys-ii/): generalization
  across model families; the two-stage protocol ("small probes guided
  search; large probes judged final candidates") adopted as our
  sweep→confirmation design; single-layer k-repeats (their layer-10×3
  math gain) adopted as our exploratory k∈{3,4} scan; the Pareto framing
  (quality delta vs compute overhead) adopted as our Δ-vs-block-size
  frontier plot. Their beam-search multi-block composition and XGBoost
  surrogate model were considered and **not** adopted: their own results
  show composition yields sublinear returns with contiguous blocks
  dominating the efficiency frontier, and our 1176-config space is fully
  enumerable, leaving nothing for a surrogate to interpolate.
- **Sapir-Whorf post** (https://dnhkng.github.io/posts/sapir-whorf/): the
  mechanistic explanation — 8 languages × 8 topics, per-layer-centered
  pairwise cosine similarity showing decode→reason→encode anatomy in five
  LLM families, middle layers organizing by content with language identity
  stripped; duplication works there because middle-layer input and output
  inhabit the same semantic manifold. Adopted wholesale as our anatomy
  experiment (§7) with style↔language, shape↔topic; also adopted:
  per-layer centering, the three-curve presentation, and the
  registered-prediction use of anatomy relative to the sweep.
- **github.com/dnhkng/RYS**: implementation is decoder-LLM-specific
  (ExLlama scanning, `model.layers.*` naming, MoE handling) — nothing
  directly reusable for a ViT encoder. Adopted: the staged pipeline shape
  (queue → scan → analyze → judge/export), mirrored in our
  sanity → anatomy → smoke → sweep → confirm → report runbook. Their
  `hf_export` step (materializing a winning config as a checkpoint with
  physically duplicated layers) is noted as future work should any config
  survive confirmation.

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

**Why not a multimodal model (and why "bigger" VLMs don't help).** Vision-
language models were excluded — originally InternVL-78B, later re-examined
for Qwen2.5-VL-72B — for two reasons. (a) *Attribution:* duplicating
vision-tower layers inside a VLM and measuring task behavior confounds the
visual representation change with the LLM's tolerance to perturbed visual
tokens; duplicating decoder layers is just original RYS on an LLM (dnhkng's
Qwen2-72B result). (b) *Scale is in the wrong place:* Qwen2.5-VL-72B's
vision encoder is a ~675M-parameter ViT (32 layers, width 1280) held
constant across the 3B/7B/72B family — the "72B" is the language decoder —
so as a *vision* transformer it is ~26× smaller than EVA-CLIP-18B's
encoder. Google's ViT-22B, the only larger dense ViT ever trained, has no
public weights.

**Structural hypothesis noted in advance (ViT vs LLM phase structure).**
An LLM plausibly needs all three phases — encode (tokens → abstractions),
reason, decode (abstractions → token space) — because its output must
return to the input's format. A vision *encoder's* output IS an embedding:
there is no format to return to, so a ViT may lack a decode phase
entirely, and any content-dominant band could extend to the final layer
rather than closing ~15 layers before the end as in LLMs. Corollary: if
duplication tolerance exists, it might persist through the last layers of
a ViT — a qualitative structural difference from the LLM heatmaps worth
checking explicitly. (The Sapir-Whorf post's observation that
encode/decode blocks are ~15 layers each, with RYS failing on models too
shallow to have a distinct middle, is the LLM-side baseline for this
comparison.)

### 2.1 Second arm: DINOv3 ViT-7B/16 (objective ablation)

**facebook/dinov3-vit7b16-pretrain-lvd1689m** (Meta, 2025): ~7B parameters,
40 blocks → 820 duplication configs, hidden 4096, patch 16 at 224×224 →
201 tokens (CLS + 4 register tokens + 196 patches), RoPE position encoding,
trained by image-only self-supervised distillation on 1.7B images
(LVD-1689M) — **no text supervision anywhere**.

*Rationale.* The anatomy caveat for EVA-CLIP (§7.4) is that a
contrastively trained model may legitimately treat rendering style and
background as content, because captions describe them. DINOv3 is the
largest open-weights ViT with a purely visual objective, so running the
**identical experiment** (same datasets, probe sets construction, splits,
noise control, sweep, k-repeats, confirmation, anatomy) on it turns that
caveat into a test: a content band in DINOv3 but not EVA-CLIP implicates
the training objective; a band in neither generalizes the negative result
across objectives. Model is the ONLY variable changed between arms.

Arm-specific implementation facts (verified: the scanner's manual forward
reproduces the model's own `forward` to 0 ulp on a random-init config):

- transformers-native `DINOv3ViTModel` (no remote code); processor via
  `AutoImageProcessor` from the same repo. Requires `transformers ≥ 4.56`,
  which conflicts with the EVA arm's `< 4.50` pin → **each arm runs in its
  own virtualenv** (`.venv` / `.venv-dino`) with the same experiment code.
- Blocks at `model.layer`, called as `layer(h, None, rope)`; RoPE tensors
  are computed once per input by `rope_embeddings(pixel_values)` and shared
  by all layers (the scanner's embed hook stashes them; they depend only on
  the fixed 224×224 grid).
- **Feature space: final `norm` + CLS** — the model's own pooling path,
  mirroring the post-LN-CLS decision of the EVA arm.
- Patch pooling in the anatomy skips `num_prefix_tokens = 5` (CLS + 4
  register tokens); EVA uses 1. Set per model by the registry.
- Gated repo (Meta license acceptance required per HF account).
- *Honest prior, recorded before the DINOv3 anatomy runs:* DINO-family
  patch features are known to encode style/texture richly (part of why
  they excel at dense tasks), so a style-dominant anatomy would not be
  shocking here either; the point of the arm is to measure rather than
  assume, with the objective as the only changed variable.

**Arm ordering and prediction registration.** The DINOv3 arm is queued to
start automatically when the EVA arm's pipeline (sweep + confirmation +
report) completes; its anatomy runs and its prediction is recorded before
its own sweep produces results, exactly as in §7.3. Results land in
`results-dinov3/` + `outputs-dinov3/`, fully parallel to the EVA arm's
`results/` + `outputs/`.

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

### 8.1 Data-retention policy (full plot-regeneration guarantee)

Every figure and table is derivable offline from persisted artifacts — no
GPU time, recomputation, or memory of the run is needed to regenerate,
restyle, or combine plots. Per arm (`results/`, `results-dinov3/`):

| Artifact | Contents |
|---|---|
| `{ds}_{split}_{accuracy,mrr}_matrix.npy` | full (L, L+1) score matrices, all 4 split×metric combinations, per dataset |
| `{ds}_percase.npz` | per-image correctness bitmap + true-class rank for every (config, split), plus the config index |
| `{ds}_baseline_features.npz` | baseline embeddings + labels for reference/borderline/random sets (enables re-running noise controls, baseline per-image outcomes, any new probe variant in baseline space) |
| `probe_sets.json` | every selected image index (refs, candidates, borderline, random) — exact probe-set reconstruction |
| `probe_meta.json` | class counts, per-dataset baselines, full noise-control distributions |
| `baseline_results.json` | baseline scores per dataset × split |
| `scan_log.jsonl` | append-only per-config score log with timestamps implicit in order |
| `{ds}_repeats.json` | k-repeat scores per (layer, k) × split |
| `aggregated_{split}_{metric}_{delta,zscore}.npy` | aggregate matrices (also recomputable from the per-dataset files) |
| `confirm_results.json` | stage-2 per-config deltas, CIs, control statistics, per-dataset breakdown |
| `confirm_percase.npz` | stage-2 per-image outcomes (config + baseline), fresh-test indices/labels, resampled reference indices, per dataset |
| `anatomy/anatomy.json` | all similarity curves (4 categories × 2 poolings) + content-dominant layer sets |
| `anatomy/anatomy_features_{pool}.npz` | raw per-layer pooled features (fp16, (L+1, 128, D)) + content/style labels — any alternative pairing, pooling comparison, or PCA view is recomputable |

Synthetic stimuli (including the anatomy grid) are seed-deterministic, so
image-level artifacts are reproducible from code + `probe_sets.json`
without storing images. Results are additionally rsynced off the GPU box to
local storage at monitoring intervals and at completion.

## 9. Presentation conventions

Upper-triangular heatmaps identical in convention to the RYS posts (y = i,
x = j, RdBu_r diverging colormap centered at 0, red = improvement, top-5
annotated), per dataset × split × metric, plus raw and z-scored aggregates;
Δ-vs-block-size Pareto scatter with frontier (RYS-II convention);
single-layer k-repeat curves labeled "exploratory — no CIs"; anatomy
three-curve plot (+ ceiling curve) with the content-dominant region shaded,
and per-layer PCA panels.

## 10. Considered and rejected (design decisions by exclusion)

- **Linear probes** (logistic regression on frozen features): a trained
  probe can partially compensate for a degraded embedding space, masking
  exactly the effect under study; also adds training variance and cost ×
  1176 configs. Nearest-centroid is training-free and cannot adapt.
  (`probes/linear_probe.py` retained in the repo but unused.)
- **Beam-search multi-block composition / surrogate model** (RYS-II): see
  §0 — sublinear returns in the source work; enumerable config space.
- **Multimodal models**: see §2 — attribution confound + vision towers are
  small even in 72B-class VLMs.
- **Wider anatomy jitter** (e.g. ±45° rotation): would demand stronger
  invariance but corrupts content labels themselves (a square at 45° is a
  diamond); ±15° kills pixel matching while keeping shape identity
  unambiguous, and the style axis — not jitter — carries the
  appearance-variation load.
- **Coarse config grids / sampling**: a step-4 grid would miss 2–3-layer
  circuits entirely; exhaustive scanning makes heatmap structure
  interpretation-free.
- **CLEVR and additional fine-grained natural-photo datasets** (Food-101,
  Pets, Flowers): inherited exclusions from the original plan — redundant
  cognitive operations, weaker confound control than the custom synthetic
  probes.

## 11. Future work (recorded during design, not part of this experiment)

- **Behavioral validation in a VLM**: apply the best/worst configs from
  the encoder sweep to a VLM's vision tower (e.g. Qwen2.5-VL) and measure
  VQA/captioning behavior — tests whether representation-level duplication
  tolerance survives contact with a downstream consumer of the tokens, and
  addresses limitation #1. Kept out of the present design for the
  attribution reasons in §2.
- **Checkpoint materialization** of any confirmed config (RYS-repo-style
  `hf_export`), yielding a deployable duplicated-layer model.
- **Foreground-masked anatomy pooling**: the renderer provides exact shape
  masks; pooling only foreground patches would give a third anatomy view
  immune to the background critique of mean-patch pooling.
- **Natural-image anatomy stimuli** (e.g. photo/sketch/painting domains of
  the same object classes) to complement the procedural grid.

## 12. Phase 1 empirical record (EVA arm, 2026-07-21)

Baseline nearest-centroid accuracy (candidates / borderline split / random
split) and the noise-control null, as measured before any duplication
config was scored:

| dataset | classes | refs | cand | borderline | rand | max noise Δacc |
|---|---|---|---|---|---|---|
| imagenet-100 | 100 | 500 | 0.925 | 0.500 | 0.960 | +0.002 |
| eurosat | 10 | 150 | 0.899 | 0.500 | 0.930 | +0.000 |
| stanford40 | 40 | 200 | 0.862 | 0.500 | 0.920 | +0.008 |
| inside_outside | 2 | 100 | 0.839 | 0.500 | 0.900 | +0.009 |
| spatial_relation | 4 | 152 | 0.818 | 0.500 | 0.800 | +0.004 |
| same_different | 2 | 100 | 0.711 | 0.500 | 0.820 | +0.007 |
| fgvc_aircraft | 100 | 500 | 0.691 | 0.500 | 0.730 | +0.000 |
| dtd | 47 | 235 | 0.684 | 0.500 | 0.750 | +0.006 |
| places365-100 | 100 | 500 | 0.587 | 0.500 | 0.600 | +0.000 |
| counting | 10 | 150 | 0.392 | 0.500 | 0.390 | +0.000 |
| symmetry | 2 | 100 | 1.000 | 1.000 | 1.000 | +0.000 |

Notes for interpretation:

- Borderline splits landed at exactly 0.500 everywhere stratification was
  possible, and the noise null is ≤ +0.9pp on all datasets — empirical
  confirmation that the selection debiasing behaves as designed.
- **Symmetry is at ceiling** (100% on 1000 candidates): EVA-CLIP separates
  mirrored from non-mirrored dot patterns perfectly, so this probe is
  one-sided for this model — it can only detect degradation, its borderline
  split is degenerate (no misclassified candidates existed; the split
  contains only correct images at 1.000 baseline), and it contributes no
  upside signal to aggregates. A notable capability observation in its own
  right.
- Counting at 0.392 (chance 0.10) shows genuine but far-from-ceiling
  numerosity signal — the most headroom of any probe in both directions.

### 7.5 DINOv3 anatomy outcome (recorded 2026-07-22, pre-sweep)

**The objective ablation split cleanly.** DINOv3-7B (image-only
self-supervision) HAS a content-dominant band; EVA-CLIP-18B (text
supervision) does not:

- CLS pooling: content > style at layers **35–40 of 40** (max gap +0.018 at
  L39; final-layer gap +0.017 — the band extends to the network's END).
- Mean-patch pooling: layers **28–40** (13 layers, max gap +0.015 at L36).

Two conclusions registered before the DINOv3 sweep completes:

1. The absence of a content band in EVA-CLIP is attributable to the
   **training objective** (caption-matching keeps appearance information
   dominant in representation geometry), not to vision-transformer
   architecture per se.
2. The DINOv3 band **runs to the final layer** — the direct signature of
   the no-decode-phase hypothesis (§2): unlike LLMs, whose content phase
   closes ~15 layers before the output to re-enter token space, a vision
   encoder's content-dominant region has nowhere it needs to return to.
   (Gap magnitudes are modest (~+0.02 centered cosine) vs the LLM results —
   style remains strongly represented throughout.)

**Registered prediction for the DINOv3 sweep:** duplication most tolerated
(and any positive effects concentrated) within/near the band, i.e. blocks
inside layers ~28–40; early-layer duplication harmful, as in EVA.

Auxiliary Phase 1 observation (arm 2): DINOv3 baselines on the geometric
synthetic probes are dramatically higher than EVA's (inside/outside 0.994
vs 0.839; symmetry 0.991 vs 1.000-at-ceiling; both near ceiling for
DINOv3) — self-supervised dense features are markedly better at these
geometry tasks, so those probes are near-ceiling (degradation-sensitive
only) in arm 2, and their borderline splits sit above 0.9 rather than 0.5.

### 12.1 EVA arm completion record (2026-07-22)

Full pipeline completed without incident: 11 datasets × 1176 configs × 2
splits (sweep), single-layer k-repeats, stage-2 confirmation, figures,
report. Depth signature unanimous across all 11 datasets on the random
split: early-layer duplication harmful (−3 to −16pp regional means,
worst on DTD textures), mid-stack near-free (0 to −2pp), late-stack
essentially free (≈0pp; mildly positive on counting +0.7pp and FGVC
+0.1pp) — consistent with the registered no-content-band prediction in its
depth-structured form, and with the no-decode-phase hypothesis (late-layer
immunity is the opposite of the LLM finding).

**Stage-2 confirmation verdict: 0 of 20 top configs met the pre-specified
criterion** (bootstrap CI > 0 AND Δ > control mean + 2·SD). The report
accordingly declares no confirmed improvement. Two facts to report
together, without post-hoc criterion changes:

1. Random-config controls: mean −4.4pp, SD 5.9pp → null band +7.3pp. The
   SD is inflated by the *harmful* left tail of random configs (early-layer
   controls at −10 to −20pp); as a symmetric band around a negative mean it
   is a very conservative bar for small positive effects.
2. All 12 leading top configs nevertheless reproduced small positive deltas
   on fresh images with 95% CIs excluding zero (+1.2 to +2.0pp, e.g.
   (34, 41): +2.0pp CI [+1.3, +2.7]) — a selection-free replication of a
   small real effect concentrated in the late-middle region (i ≈ 24–34,
   j ≈ 41–48). Notably (34, 41) is a 7-layer block — the same block size as
   the original RYS optimum on Qwen2-72B.

Interpretation for the paper: no config produces improvements that are
large relative to what an arbitrary duplication can do by luck (the
pre-specified test), but the late-middle region shows a small, replicable
positive effect (~+1–2pp) that is real by the paired-bootstrap standard.
Frame as "weak but reliable second-pass benefit in the late-middle stack;
no LLM-scale reasoning-circuit gains," and discuss the control-band
construction explicitly.

### 12.2 DINOv3 arm completion record (2026-07-22)

Full pipeline completed (~6h; one relaunch after a torchvision build
mismatch in the arm's venv, before any results existed). Random-split
results, all 11 datasets:

- **Dramatically more duplication-tolerant than EVA overall**: whole-matrix
  mean deltas −0.1 to −1.1pp (EVA: −3 to −10pp). Early-layer fragility
  remains unanimous (−5 to −14pp regional means); everything from
  mid-stack onward is ≈0.
- **Stage-2 confirmation: 0 of 20 top configs met the pre-specified
  criterion.** Controls: mean −0.05pp, SD 0.47pp (null band +0.88pp — tight
  because random duplications barely hurt this model). Leading top configs
  reproduced tiny positive deltas with CIs excluding zero (+0.5 to +0.8pp),
  clustered at (i ≈ 13–21, j ≈ 24–31). One of 20 random controls exceeded
  the band (expected false-positive count ≈ 0.5) — consistent with noise.
- Several probes are at/near ceiling for DINOv3 (baselines: imagenet-100
  0.99, spatial 0.99, inside/outside 1.00, symmetry 1.00, stanford40 0.96)
  — upside structurally invisible there; the informative datasets were
  counting (0.42), places365 (0.61), dtd (0.75), fgvc (0.73).

**Registered-prediction scorecard (from §7.5):**

1. "Early-layer duplication harmful" — ✓ confirmed, both arms, all datasets.
2. "Duplication tolerated within the band (28–40)" — ✓ (band deltas ≈ 0.000
   everywhere), but tolerance is NOT band-exclusive: mid-stack (13–27) is
   equally tolerated. The anatomy band under-predicts the extent of the
   tolerant region.
3. "Positive effects concentrated in/near the band" — ✗: the (tiny,
   unconfirmed) positive effects concentrate in mid-stack (13–31), largely
   BELOW the band. Content-dominance locates tolerance imperfectly and
   locates benefit poorly.

**Cross-model summary for the paper:** RYS-style duplication gains do not
transfer to ViT representation quality in either training regime — no
config in either arm survives selection-robust confirmation. The
depth-structure story is the positive contribution: (a) early layers are
universally fragile; (b) late layers are universally immune (opposite of
LLMs; the no-decode-phase signature, corroborated by DINOv3's content band
running to the final layer); (c) the content band exists only without
language supervision (objective ablation, §7.5); (d) small CI-positive
second-pass effects exist in both models (+1–2pp EVA late-middle, +0.5–0.8pp
DINOv3 mid) but are modest relative to what arbitrary-config luck can
produce, per the pre-specified control-band test.

## 13. Known limitations (to state in the paper)

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
