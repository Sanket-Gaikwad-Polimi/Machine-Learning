# Design decisions

We wrote these decisions down while building the project (2026-10-25), before the test set was scored. Post-hoc findings are marked as such. Together with the rules in README.md (section 9), they show that no choice was tuned on test results.

## Brightness (from the EDA)
- Good test images are darker (mean ~0.67) than the training images (~0.72) and the defective test images (~0.73): risk of false positives.
- src/data.py has an optional per-image brightness normalisation flag, OFF by default, decided a priori, never tuned on test results.
- Robustness step: check score stability on the 40 validation images under brightness changes.
- Report both models with and without normalisation.

## Resolution
- Main results at 256 px. A larger size only as a reported comparison, never chosen by test scores. (In the end we did not run a larger size.)

## Thresholds (src/metrics.py, fixed a priori)
- Image threshold T_image = 99th percentile of the 40 validation scores; pixel threshold T_pixel = 99.9th percentile of all validation heat-map pixels.
- Caveat: with only 40 validation images, T_image lies between the two highest validation scores and depends on which images were drawn.

## Autoencoder (src/autoencoder.py, fixed a priori)
- 1 grey channel in [0, 1], no ImageNet normalisation, 256 px, bottleneck 32 x 8 x 8, MSE loss, Adam (lr 1e-3), 120 epochs, batch 8, seed 42.
- Training augmentation: random rotation (corners filled with the image's median grey) and flips.
- Best epoch = lowest loss on the 40 validation images.
- Heat-map = per-pixel squared error, Gaussian sigma 4. Score = maximum of the smoothed heat-map.
- Checkpoint: outputs/checkpoints/autoencoder.pt (autoencoder_bn.pt with brightness normalisation).

## PatchCore (src/patchcore.py, fixed a priori)
- wide_resnet50_2 IMAGENET1K_V1, frozen. Input 256 px, grey repeated to 3 channels + ImageNet normalisation.
- layer2 + layer3, 3x3 local average pooling, layer3 upsampled to 32 x 32, concatenated: 1536 dims per patch.
- Memory bank from the 280 training images only (286,720 patches), greedy coreset to 10% (seed 42, random projection to 128 dims used only for the selection).
- Patch score = nearest-neighbour distance; image score = maximum patch score; heat-map = bilinear upsampling to 256 + Gaussian sigma 4.
- No augmentation. Brightness flag off by default. Checkpoint: outputs/checkpoints/patchcore_bank.pt (patchcore_bank_bn.pt).

## Evaluation protocol (evaluate.py, fixed before any test image was scored)
- Variants: ae_raw, ae_bn, pc_raw, pc_bn (bn = brightness normalisation on). Missing ones are built from train/val only.
- Thresholds per variant from its 40 validation images, written to outputs/results/thresholds.csv before Phase B (test).
- Robustness perturbations (applied to the 256 px images in [0, 1], the same images for all variants): brightness -0.05 / +0.05, contrast x0.8 / x1.2 around the image mean, rotation 10 deg (corners filled with median grey), Gaussian noise sigma 0.02 (seed 42 + index). Measured on validation (score change, false-reject rate) and on test (AUROC, false-reject rate, recall at the fixed threshold).
- Example figures chosen by fixed rules: highest-scoring false positive, lowest-scoring missed defect, median-score detection of each defect type.
- The test set is scored once; afterwards only reporting/plotting code may change.

## The test run (2026-10-25)
- evaluate.py ran ONCE on the 160 test images (10.3 min). Frozen results: outputs/results/metrics.csv, per_defect.csv, robustness.csv, thresholds.csv, <variant>_scores.csv, evaluate_log.txt, and the test heat-maps (published in the Release).
- A normal rerun refuses to start. `python evaluate.py --report-only` only redraws tables and figures from the saved scores. `python evaluate.py --out-dir outputs/rerun` (added in the final step) repeats the whole evaluation in a separate folder without touching the official files.
- Any further analysis uses validation images only and must not change models, thresholds or the frozen results.

## Post-hoc diagnostics (diagnostics.py, validation images only)
- Frozen files unchanged (SHA-256 checked); thresholds read from thresholds.csv; no test image loaded.
- PatchCore's rotate_10 drop comes mostly from the flat grey corner wedges (wedges_only +68.6% vs rotate_10 +72.3%); a clean 10 deg rotation at 1024 px with reflect padding gives +8.9% (8/40 validation images above T).
- Noise sigma 0.02 added at 1024 px before resizing (camera-like) has no effect; the frozen noise test added it after resizing (unnatural texture, +41.9%).
- rot90 gives +31.7% for PatchCore (hypothesis, not tested: rotated lighting/background, never seen in training). The autoencoder reacts much less to these conditions.
- The identical rotate_10 AUROC of ae_raw and pc_raw (3537 of 4879 pairs) is a coincidence, not a bug (different scores: false-reject rate 0 vs 1 at their own thresholds).
- Reported as a labelled post-hoc analysis next to the frozen robustness numbers, never as a replacement.

## Final step: README and code freeze (2026-10-25)
- README.md rewritten for teammates; PROJECT_GUIDE.md removed (its useful content is in the README).
- Trained models and test heat-maps are published as GitHub Release v1.0 (checkpoints_v1.zip), not in git.
- The code is frozen after this step.
