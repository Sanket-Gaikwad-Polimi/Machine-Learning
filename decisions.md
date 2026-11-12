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

### Phase 2: synthetic defects on the validation images (src/synthetic.py + phase2.py; written before anything was generated or scored)
Purpose: a "validation set with defects", so Phase 3 can be judged without any test image. Everything below is fixed now and is not changed after scoring.
- Source: only the 40 validation images of split.json. No test image and no test mask is used.
- Defects are made on the original 1024 px image (float in [0, 1]), then rounded to 8 bit and resized to 256 px with PIL bilinear, exactly like data.load_image. The mask is made at 1024 px and resized with nearest-neighbour, exactly like data.load_mask. Reason: diagnostics.py showed that changes made at 256 px create textures no real photo has.
- Screw (foreground) rule, fixed: pixel darker than the image's median grey minus 0.30 (the median is the background, since the screw covers about 10% of the image); then binary closing (6 iterations), opening (3), keep the largest connected region, fill holes. Every defect is placed on this region: its centre must lie on the screw shrunk by 12 px, and the image is changed only inside the screw region, so the plain background is never touched. The mask = the defect shape cut to the screw region.
- Screw axis, for the thread damage: principal axis of the screw region; position t = 0 at the head and 1 at the tip. The head is the end whose outer 20% of the length contains more screw pixels (it is wider). Threaded part = t between 0.55 and 0.95.
- 4 types (sizes in px at 1024; "sign" is + or - with probability 1/2; all values uniform in the ranges):
  1. scratch: straight line, length 60-250, width 2-6, any angle; brightness change sign x 0.06-0.15, soft edges (Gaussian blur sigma 1). Mask = the line widened by 4 px on each side (1 px at 256: like a hand-drawn mask). At least 90% of the line must lie on the screw.
  2. spot (contamination): ellipse, half-axes 15-50, any angle; brightness change sign x 0.08-0.20, strongest in the centre and fading to 0 at the ellipse border. Mask = the ellipse.
  3. dent / missing material (CutPaste-style): rectangle, sides 30-90, filled with the patch of the same size copied from another point of the same screw (centre at least 1.5 x the longer side away); edges blended (alpha blurred with sigma 3). Mask = the rectangle.
  4. thread damage (smear): ellipse, half-axes 20-50, centre on the threaded part; the image there is replaced by a Gaussian-blurred copy (sigma 3-8), fading out towards the ellipse border. Mask = the ellipse.
- Size: defect area (mask at 256 px) between 0.1% and 1% of the image. Honest note: this range comes from the EDA defect-size statistics, which were measured on the TEST masks (median 0.29%, 118 of 119 below 1%). Only this size range is taken from the test set, never any score.
- Visibility: the largest grey-value change inside the mask at 256 px must be at least 0.03 (about 8 grey levels), so no "invisible" defect is counted. A draw that fails the area, on-screw or visibility rule is drawn again with the next random numbers (at most 500 draws).
- One defect of each type on each of the 40 images: 160 synthetic defective images with exact masks. Random numbers: numpy default_rng([42, image index, type index]), so each image is reproducible on its own. The images are generated on the fly and not stored; only a parameter table (CSV) and an example figure outputs/v2/phase2/synthetic_examples.png are saved. Example rule, fixed: validation images 0 and 20, all 4 types each (8 examples), shown as the full image with the mask outline, a crop before and after, and the mask.
- The example figure is checked by eye BEFORE scoring. If the defects look unrealistic or not subtle, the generator may be changed at that point (and this section updated), but never after any image has been scored.
- Baseline (phase2.py): the 4 frozen v1 variants, frozen T_image and T_pixel from thresholds.csv, nothing retrained. Scored: the 40 good validation images (must equal <variant>_val_scores.csv) and the 160 synthetic images. Reported per variant, per type and for all 160: image AUROC (40 good vs the synthetic ones), recall at T_image, pixel AUROC and IoU / Dice at T_pixel (src/metrics.pixel_metrics, on the good + synthetic images), localisation hit rate (heat-map maximum within 8 px of the mask, as in Phase 1) with the median distance and the chance level. Outputs in outputs/v2/phase2/ (CSV, PNG, log), labelled "synthetic defects on validation images". The 40 good images are the same ones that set T_image, so the false-reject rate here is about 1/40 by construction and is not a result.
- These baseline numbers are the yardstick for Phase 3. The generator is not tuned to make the synthetic results look like the real test results; any similarity or difference is only reported as an observation.
- Hash check: every file in outputs/results, outputs/figures, outputs/checkpoints and outputs/analysis is hashed (SHA-256) before and after; all must be unchanged.
- Example figure checked by eye (2026-11-13): the generator is accepted as it is, nothing changed.
- Note on IoU: scratch masks are wider than the changed line (4 px margin on each side at 1024), so the scratch IoU / Dice is pessimistic. IoU and Dice on synthetic defects are only compared between models (and between v1 and Phase 3), never read as an absolute quality.
- Limitation, stated before Phase 3: every synthetic defect lies at least 12 px inside the screw, and it is placed with the same screw rule that a background mask would use. So this set cannot show whether a background mask cuts off real defects at the screw's edge (e.g. manipulated_front, thread_side). Any Phase 3 background mask must therefore be dilated generously, and this risk must be stated when it is proposed.
