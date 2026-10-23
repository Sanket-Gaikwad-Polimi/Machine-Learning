# Project Guide: Anomaly Detection on Screws

A short overview so all three of us share the same picture of the project.

## 1. What we are doing

- We build a system that looks at a photo of a screw and says **"good" or "defective"**, and **shows where the defect is** (a heat-map).
- We train **only on images of good screws**. The model learns what "normal" looks like and flags anything that differs.
- We compare **two methods** and answer the professor's question:
  *Do pretrained image features detect and locate defects better than an autoencoder trained only on our screws?*

## 2. Our data (`screw/` folder)

- **train/good**: 320 images of good screws. The only images we train on.
- **test/good**: 41 good screws.
- **test/<defect>**: 119 defective screws in 5 types: `manipulated_front` (24), `scratch_head` (24), `scratch_neck` (25), `thread_side` (23), `thread_top` (23).
- **ground_truth/<defect>**: one black-and-white mask per defective image; white = where the defect is.
- Images: greyscale, 1024 × 1024 px.
- Things to know about screws:
  - they lie at **random angles and positions**;
  - defects are **tiny** (often less than 1% of the image);
  - the background has **dust specks** that can cause false alarms.

## 3. Key ideas in plain words

- **One-class learning**: learn "normal" only; anything unusual gets a high score.
- **Anomaly score**: one number per image. High means probably defective.
- **Threshold (T)**: score above T means "defective". We choose T using good images only.
- **Heat-map**: a score for every pixel, showing *where* the problem is.
- **Model 1, autoencoder**: a network that compresses an image and rebuilds it. Trained on good screws, it rebuilds defects badly, and the difference |original − rebuilt| is the heat-map. (Covered in the anomaly detection slides.)
- **Model 2, pretrained features (PatchCore-style)**: a network already trained on millions of photos (ImageNet) describes every small patch of the image. We store the patches of good screws; a test patch that is far from all stored patches is anomalous. No training needed. (Not in the slides; we learn it from the PatchCore paper.)
- **Metrics**:
  - **AUROC**: how well scores separate good from defective (1.0 = perfect, 0.5 = guessing). Our main number.
  - **Precision / recall / F1, confusion matrix**: how many defects we catch vs how many good screws we wrongly reject.
  - **IoU / Dice**: how well our heat-map overlaps the true defect mask.

## 4. The golden rule

**Defective test images are never used for training, for choosing the threshold, or for tuning anything.** They are used only once, for the final evaluation. Breaking this rule makes our results invalid.

## 5. The plan, step by step

1. **Data check**: dataset is in the repo; verify the counts above.
2. **Explore**: look at good screws and every defect type with its mask; measure how big defects are.
3. **Prepare**:
   - resize images (try 256 px, then test bigger, because defects are tiny);
   - split the 320 good images into **280 train / 40 validation** once, and save the split to a file everyone uses.
4. **Model 1, autoencoder**: train on the 280 good images, then produce scores and heat-maps.
5. **Model 2, pretrained features**: build the memory bank of good-screw patches, then produce scores and heat-maps.
6. **Outputs**: for every test image, a score, a heat-map, and a predicted defect region.
7. **Threshold**: set T from the 40 good validation images only; discuss rejecting good screws vs missing defects.
8. **Evaluate**: on the 160 test images, compute AUROC, precision, recall, F1, confusion matrix, pixel AUROC, IoU and Dice for both models.
9. **Defect types**: results per defect type; which are easy, which are hard, and why. Show correct detections, false alarms and missed defects.
10. **Robustness**: change brightness, contrast, rotation or noise of the test images and see whether the models break.
11. **Demo and presentation**: `inspect("image.png")` shows the score, the good/defective decision, the heat-map and the defect overlay. Then slides with results and our answer to the main question.

## 6. Who does what

| Person | Main responsibility | Steps |
| --- | --- | --- |
| **A**: data and evaluation | Data loading, split file, metrics code, threshold logic, robustness tests | 1, 2, 3, 7, 8, 10 |
| **B**: Model 1 | Autoencoder: design, training, scores, heat-maps | 4, 6 (for Model 1) |
| **C**: Model 2 | Pretrained-feature model: feature extraction, memory bank, scores, heat-maps | 5, 6 (for Model 2) |
| **Everyone** | Defect-type analysis, demo, presentation | 9, 11 |

Person A's split file and metrics code come first, because both models depend on them.

## 7. How we work together

- Code lives on GitHub. Pull before you start, push when you finish.
- Each person works in **their own files and notebooks** (for example `notebooks/02_autoencoder_<name>.ipynb`). Don't edit someone else's notebook.
- Never type full paths like `D:\...` in code. Use paths relative to the project folder (for example `screw/train/good`), so code runs on every laptop.
- Save every figure and result table into `outputs/` so everyone can see them.
- Everyone uses the **same split file** and the **same random seed**.

## 8. What each of us should understand first

- **Everyone**: anomaly detection slides (autoencoder, heat-map, threshold); what AUROC, IoU and Dice mean.
- **Person B**: convolutional autoencoders (Conv2d, ConvTranspose2d, bottleneck size).
- **Person C**: pretrained CNNs (ResNet), feature maps, nearest-neighbour distance (PatchCore paper: https://arxiv.org/abs/2106.08265).
- **Person A**: train/validation/test splits, ROC curves, scikit-learn metrics.

## 9. What we hand in

- **Code**: this repository, runnable by following README.md.
- **Presentation** with:
  - the comparison table of both models (image level and pixel level);
  - ROC curves and the per-defect-type results;
  - example heat-maps, including false alarms and misses;
  - the robustness results;
  - our answer to the main question.
- **Individual theory exam** on the same day, so everyone must understand both models, not only their own part.
