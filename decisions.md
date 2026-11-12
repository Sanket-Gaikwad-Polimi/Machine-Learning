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

## v2: post-hoc improvements (started 2026-11-12)
Everything below comes after the v1 test results were seen. It never replaces v1.
- v1 stays frozen: outputs/results/*, outputs/figures/eval_*, diag_*, the checkpoints and the thresholds are never modified. Every v2 script proves it with SHA-256 hashes before and after, like diagnostics.py.
- Each v2 decision is written here BEFORE the experiment it governs runs.
- Plan: Phase 1 analysis of v1 -> Phase 2 synthetic defects on validation images -> Phase 3 improvements judged ONLY on validation + synthetic defects -> Phase 4 freeze v2, run it once on screw (labelled "post-hoc: screw test set seen before"), and run v1 and v2 once on a second category (transistor), whose test set nobody has scored yet.

### Phase 1: analysis of the frozen v1 results (analysis.py; written before it was run)
- Reads saved files only: <variant>_scores.csv, <variant>_val_scores.csv, thresholds.csv and metrics.csv (outputs/results), the test heat-maps outputs/checkpoints/<variant>_heatmaps.npz, and the masks (256 px, nearest-neighbour). No model is loaded and nothing is re-scored. Outputs go to outputs/analysis/ (CSV + PNG), labelled "post-hoc analysis of frozen v1 results". Self-check first: AUROC, recall, false-reject rate and IoU recomputed from these files must equal metrics.csv.
- Localisation, all 4 variants, the 119 defective test images: peak = position of the heat-map maximum at 256 px (the first one if tied). Distance = Euclidean distance in pixels at 256 px from the peak to the nearest defect pixel (0 inside the mask). Hit = distance <= 8 px, i.e. the peak lies in the mask dilated by a disk of radius 8 px (one PatchCore cell). Radius fixed a priori. Reported: hit rate overall, per defect type, and for detected images only (TP at the frozen T_image). For context, the chance level = mean share of the image covered by the dilated mask (the hit rate of a randomly placed peak).
- Threshold trade-off, hypothetical only (the official T stays): ROC and precision-recall curves from the test scores with the frozen T marked. Recall at test false-reject rates of at most 0%, 2.5%, 5% and 10%: with 41 good test screws that allows 0, 1, 2 and 4 false rejects, and the hypothetical T is the (k+1)-th highest good test score (decision: score > T). Validation-only alternative rules: T = max of the 40 validation scores; T = mean + 3 std of the 40 validation scores (sample std, ddof 1); their test recall and false-reject rate shown as what-if.
- Uncertainty: bootstrap with 2000 resamples, seed 42 (numpy default_rng), stratified: the 41 good and the 119 defective test images are resampled with replacement separately. The same resampled images are used for all 4 variants (paired). Metrics at the frozen thresholds: image AUROC, recall, false-reject rate, IoU (mean per-image IoU over the resampled defective images, as in metrics.csv). 95% CI = 2.5th and 97.5th percentiles. Paired bootstrap of the AUROC difference pc_raw - ae_raw, with its CI and the share of resamples where it is <= 0.
