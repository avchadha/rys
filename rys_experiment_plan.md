# Vision Transformer RYS Experiment Plan

## Goal

Apply the **RYS probing / layer-loop analysis technique** from David's
blog post to **Vision Transformer (ViT) models**.

Final outputs should include:

-   Recreated **RYS-style probing infrastructure**
-   Heatmaps showing **behavior changes across layers**
-   Image-based probe tasks
-   Analysis comparing **vision model internal behavior**

Reference blog post: https://dnhkng.github.io/posts/rys/

------------------------------------------------------------------------

# Important Rules

-   Do **not run heavy computation locally**.
-   Use **cloud compute** if needed (RunPod, Modal, Lambda Labs, Colab,
    etc.).
-   Store results and code in a **GitHub repository**.
-   Commit frequently.
-   After every major step:
    -   Run tests
    -   Verify outputs
    -   Save intermediate results

------------------------------------------------------------------------

# Phase 1 --- Understand the RYS Method

## Step 1 --- Read and Summarize the RYS Method

Tasks:

-   [ ] Carefully read the blog post
-   [ ] Identify the core components of the RYS method
-   [ ] Identify how the probing mechanism works
-   [ ] Identify how the heatmaps are generated
-   [ ] Write a concise summary

Create file:

docs/rys_summary.md

Verification:

-   [ ] Confirm the summary clearly explains the scanner method
-   [ ] Commit the summary to GitHub

------------------------------------------------------------------------

## Step 2 --- Locate the RYS Model Code

Tasks:

-   [ ] Search HuggingFace for **RYS models**
-   [ ] Identify repository for **RYS-XLarge**
-   [ ] Inspect the model card and architecture
-   [ ] Determine whether any custom code is available

Create file:

docs/rys_model_sources.md

Verification:

-   [ ] Attempt loading the model using `transformers`
-   [ ] Run a minimal inference test
-   [ ] Confirm model loads successfully
-   [ ] Commit findings

------------------------------------------------------------------------

# Phase 2 --- Recreate the RYS Scanner

## Step 3 --- Implement Layer Hooking

Tasks:

-   [ ] Load a medium-size open LLM for testing
-   [ ] Implement forward hooks
-   [ ] Record hidden states

Create file:

scanner/layer_hooks.py

Verification:

-   [ ] Print tensor shapes for each layer
-   [ ] Confirm layer order is correct
-   [ ] Run `python test_layer_hooks.py`
-   [ ] Ensure hidden states are captured

------------------------------------------------------------------------

## Step 4 --- Implement the Layer Loop Scanner

Tasks:

-   [ ] Implement hidden-state swapping or injection
-   [ ] Run model with modified hidden states
-   [ ] Measure output differences

Create file:

scanner/layer_loop.py

Verification:

-   [ ] Confirm outputs change when layers are modified
-   [ ] Log `(i, j)` combinations
-   [ ] Save matrix to `results/layer_scan.npy`
-   [ ] Run `python test_layer_loop.py`

------------------------------------------------------------------------

## Step 5 --- Implement Probe Tasks

Tasks:

-   [ ] Implement a probe interface
-   [ ] Create at least two probe types
-   [ ] Implement scoring functions

Files:

probes/math_probe.py\
probes/reasoning_probe.py

Verification:

-   [ ] Run `python run_probes.py`
-   [ ] Ensure probes return numeric scores

------------------------------------------------------------------------

## Step 6 --- Generate Heatmaps

Tasks:

-   [ ] Convert `(i, j)` scan matrix to heatmaps
-   [ ] Use matplotlib or seaborn
-   [ ] Save PNG outputs

File:

visualization/heatmap.py

Output:

outputs/heatmap.png

Verification:

-   [ ] Run visualization script
-   [ ] Confirm axes correspond to layer indices
-   [ ] Confirm the image renders correctly

------------------------------------------------------------------------

# Phase 3 --- Vision Transformer Experiments

## Step 7 --- Select a Large Vision Transformer

Tasks:

-   [ ] Search HuggingFace for large ViT models
-   [ ] Compare parameter sizes
-   [ ] Test model loading

Candidates:

-   google/vit-g
-   openai/clip-vit-large
-   facebook/dinov2-large
-   vit-huge

Create file:

docs/vit_model_selection.md

Verification:

-   [ ] Run `python test_vit_loading.py`
-   [ ] Confirm model loads successfully
-   [ ] Confirm inference works on one image

------------------------------------------------------------------------

## Step 8 --- Implement Vision Layer Hooks

Tasks:

-   [ ] Adapt layer hook system to ViT
-   [ ] Capture patch token embeddings
-   [ ] Confirm layer counts

File:

vit_scanner/vit_hooks.py

Verification:

-   [ ] Run `python test_vit_hooks.py`
-   [ ] Confirm activations captured
-   [ ] Confirm tensor shapes correct

------------------------------------------------------------------------

## Step 9 --- Create Image-Based Probes

Tasks:

-   [ ] Create probe dataset
-   [ ] Implement scoring metrics
-   [ ] Implement probe runner

File:

vit_probes/image_probe.py

Verification:

-   [ ] Run `python test_image_probes.py`
-   [ ] Confirm dataset loads
-   [ ] Confirm probes produce numeric scores

------------------------------------------------------------------------

## Step 10 --- Run Vision Layer Scan

Tasks:

-   [ ] Integrate scanner with ViT
-   [ ] Run `(i,j)` scan
-   [ ] Store results

Output:

results/vit_layer_scan.npy

Verification:

-   [ ] Confirm matrix dimensions match layer count
-   [ ] Confirm results are non-uniform

------------------------------------------------------------------------

## Step 11 --- Generate Vision Heatmaps

Tasks:

-   [ ] Generate heatmaps for each probe
-   [ ] Save visualizations

Outputs:

outputs/vit_heatmap_object.png\
outputs/vit_heatmap_color.png\
outputs/vit_heatmap_shape.png

Verification:

-   [ ] Confirm heatmaps render
-   [ ] Confirm axes correspond to layer indices

------------------------------------------------------------------------

# Phase 4 --- Analysis

## Step 12 --- Interpret Results

Tasks:

-   [ ] Analyze heatmap patterns
-   [ ] Identify critical layers
-   [ ] Compare with LLM results

Write report:

docs/analysis.md

Verification:

-   [ ] Ensure all figures referenced
-   [ ] Ensure reproduction instructions included

------------------------------------------------------------------------

# Final Repository Structure

scanner/\
vit_scanner/\
probes/\
vit_probes/\
visualization/\
outputs/\
results/\
docs/

Final deliverables:

-   RYS scanner implementation
-   Vision transformer probing framework
-   Heatmap visualizations
-   Written analysis
