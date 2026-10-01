# AMME5710-Assignment-2

AMME5710 Computer Vision and Image Processing - Assignment 2 (20%).

- **Question 1 (50%)** - 3D reconstruction using underwater stereo vision -> `MainQ1.py` (done: stereo pipeline following the Week 5 tutorial structure + detector/parameter comparison)
- **Question 2 (50%)** - Scene classification -> `MainQ2.py` + `assign2_sceneclassifier.py` (done: hand-crafted colour/texture features + SVM, following the Week 7 tutorial structure; 0.963 CV / 0.971 test accuracy)

Detailed explanation of how the code works:
- Q1 (reconstruction, outlier rejection, before/after results, functions): [`Q1_EXPLANATION.md`](Q1_EXPLANATION.md)
- Q2 (features, model selection, evaluation, functions): [`Q2_EXPLANATION.md`](Q2_EXPLANATION.md)

Report drafts and remaining to-do lists: [`Q1_REPORT_DRAFT.md`](Q1_REPORT_DRAFT.md), [`Q2_REPORT_DRAFT.md`](Q2_REPORT_DRAFT.md)

---

## Repository layout

| Path | Contents |
|---|---|
| `AMME5710_Assignment2_2026.pdf` | Assignment specification |
| `MainQ1.py` | Question 1 solution script |
| `Q1_EXPLANATION.md` | How the Q1 code works: reconstruction, outlier rejection, before/after results, function reference |
| `figures_q1/` | Figures produced by `MainQ1.py` (for the report) |
| `Q1_REPORT_DRAFT.md` | Draft text for the Q1 report (Introduction, Methodology, Results and Discussion, appendix plan, references, AI statement) and the remaining Q1 to-do list |
| `MainQ2.py` | Question 2 script: runs all experiments, evaluates, saves the model and figures |
| `assign2_sceneclassifier.py` | Q2 feature extraction + the required `assign2_sceneclassifier(image_path)` function |
| `assign2_scene_model.pkl` | Q2 trained model (created by `MainQ2.py`, loaded by `assign2_sceneclassifier`) |
| `Q2_EXPLANATION.md` | How the Q2 code works: features, model selection, evaluation, function reference |
| `figures_q2/` | Figures produced by `MainQ2.py` (for the report) |
| `Q2_REPORT_DRAFT.md` | Draft text for the Q2 report (Introduction, Methodology, Results and Discussion, appendix plan, references) and the remaining Q2 to-do list |
| `assignment2_stereodata/` | Q1 data: 49 stereo pairs + calibration / pose / terrain pickles |
| `assignment2_places/` | Q2 data: 7 scene classes x 50 images |
| `AMME5710_week*_tutorial.ipynb` | Tutorial notebooks (week 5 = stereo vision, week 7 = image classification reference) |
| `.gitignore` | Ignores `__MACOSX/`, `.DS_Store`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/` |

---

## Question 1 - what is required

Data: 49 stereo image pairs from a downward-looking diver rig over a coral reef
(left camera = colour, right camera = monochrome), plus:

- `calib_stereo_diver.pkl` - stereo calibration (`Kl, Dl, Kr, Dr, R, t`).
  Note: the PDF calls this file `stereo_calib_diver.pkl`, but the real file name is `calib_stereo_diver.pkl`.
- `camera_pose_data.pkl` - per-frame rotation `R` (3x3x49), translation `t` (3x49) and filenames.
- `terrain_data.pkl` - reference terrain `height_grid`, `X`, `Y`.

Tasks:
1. Extract interest points (ORB, SIFT, ...) in each pair and match them left/right, with outlier rejection.
2. Triangulate the matches into 3D points in the left camera frame.
3. Rotate/translate each pair's points into the world frame using the pose data and merge into one point cloud.
4. Compare the point cloud with the reference terrain.

Deliverables:
- **Code (20 marks):** `MainQ1.py` with `images_left_dir` and `images_right_dir` variables at the top; it must run and plot the point cloud. Code must be commented.
- **Report (30 marks, 4-page limit excl. appendix):**
  - Introduction (5) - how stereo vision works, applications, algorithms used.
  - Methodology (10) - features extracted/matched, world point cloud built correctly.
  - Results & Discussion (15) - compare different interest point types and parameters; effect on accuracy, density and types of structures covered.

General submission rules: one report (sections "Question 1", "Question 2", appendix), code zipped as `SID_Assignment2.zip`,
and an **AI-use acknowledgement statement at the end of the report**.

---

## Question 1 - how to run

```bash
python MainQ1.py
```

Requirements: `numpy`, `opencv-python` (>= 4.4, for SIFT), `matplotlib`.
Tested with Python 3.7 / OpenCV 4.13 (the `CleanerS` conda env; the base conda env has no numpy/cv2):

```bash
~/miniconda3/envs/CleanerS/bin/python MainQ1.py
```

Running the file does everything: the main reconstruction, the stage-by-stage outlier-rejection table, the
detector/parameter comparison, the structure (terrain slope) and CLAHE dark/bright analyses, and saving all
figures to `figures_q1/`. Results are printed to the terminal. Runtime: ~8 min. Figures open in windows as they are drawn: the 4 main-reconstruction
figures after ~1 min, the 5 comparison figures at the end. Each figure is saved as soon as it is drawn. The
first 4 windows look frozen while the comparison is computing; they respond again at the end. Closing all windows
ends the script.

Settings at the top of `MainQ1.py`:

| Variable | Default | Meaning |
|---|---|---|
| `images_left_dir`, `images_right_dir` | `assignment2_stereodata/...` | Image folders (pickles are read from their parent folder) |
| `CONFIG` | SIFT, CLAHE, ratio 0.8 | The pipeline used for the main reconstruction |
| `COMPARISON_CONFIGS` | 8 variations | `CONFIG` with the detector or a parameter changed |
| `EPIPOLAR_THRESH` | `2.0` px | Max Sampson distance to the calibrated epipolar line |
| `REPROJ_THRESH` | `2.0` px | Max triangulation reprojection error |
| `DEPTH_RANGE` | `(0.3, 6.0)` m | Plausible camera-to-seafloor distance |
| `EXAMPLE_PAIR` | `20` | Pair used for keypoint / match plots |
| `N_DRAW_MATCHES` | `150` | Matches drawn per match plot |
| `FIGURE_DIR` | `'figures_q1'` | Where figures are saved; old `q1_fig*.png` files are deleted first |

---

## Question 1 - method summary

The code follows the structure of the **Week 5 tutorial** stereo pipeline. For each stereo pair:

1. **Pre-process:** left image to grayscale; CLAHE on both images (colour vs mono sensor, low-contrast water).
2. **Detect features:** SIFT `detectAndCompute` (ORB and AKAZE in the comparison).
3. **Match:** brute-force k-NN (k=2) + Lowe ratio test (0.8).
4. **Undistort** the matched points: `cv2.undistortPoints(pts, K, d, None, K)`.
5. **Epipolar outlier rejection:** F from the known calibration (`F = Kr^-T [t]x R Kl^-1`), Sampson distance < 2 px.
6. **Triangulate:** `cv2.triangulatePoints` with `Pleft = Kl[I|0]`, `Pright = Kr[R|t]`, then divide by the 4th coordinate.
7. **Filter:** reprojection error < 2 px in both views, depth 0.3-6 m.
8. **World frame:** the pose maps world -> camera (`x_cam = R X_world + t`, Week 5 convention), so
   `X_world = R^T (x_cam - t)`. Verified: camera centres `-R^T t` fall over the terrain, about 2.5 m above it,
   with the optical axis pointing down (+Z).

All 49 pairs are merged into one point cloud.

**Terrain comparison:** each point's Z is compared with the reference surface directly below it
(`Z = -height_grid` at `(flip(X), Y)`, the layout used by the plotting code in the PDF; without the flip the
RMSE is 0.29 m instead of 0.038 m, confirming the layout).

---

## Question 1 - results

| Config | Points | RMSE | MAD | Within 5 cm | Coverage (5 cm cells) |
|---|---|---|---|---|---|
| **SIFT (`CONFIG`)** | **466k** | **0.038 m** | **0.010 m** | **93.4%** | **43%** |
| SIFT ratio 0.6 | 359k | 0.034 m | 0.009 m | 94.6% | 42% |
| SIFT no ratio test | 529k | 0.075 m | 0.010 m | 92.0% | 44% |
| SIFT no CLAHE | 191k | 0.037 m | 0.009 m | 93.3% | 38% |
| SIFT contrastThreshold 0.01 | 572k | 0.041 m | 0.010 m | 92.6% | 44% |
| ORB 1000 | 23k | 0.045 m | 0.018 m | 86.3% | 16% |
| ORB 10000 | 223k | 0.040 m | 0.018 m | 87.2% | 31% |
| AKAZE default | 329k | 0.037 m | 0.010 m | 92.3% | 41% |
| AKAZE threshold 1e-4 | 665k | 0.041 m | 0.011 m | 91.2% | 43% |

MAD = median absolute deviation (robust spread). Main SIFT cloud: median height error -0.001 m (no systematic bias).

Discussion points for the report:
- **Ratio test is key for accuracy:** without it, about half the candidate matches are wrong; the geometric filters
  remove most of them, but RMSE still doubles (0.038 -> 0.075 m). See `q1_fig07.png`.
- **CLAHE is key for density:** 2.4x more points (191k -> 466k) and coverage 38 -> 43% at the same accuracy;
  points from dark image regions increase 4.7x vs 2.0x in bright regions.
- **Accuracy vs density trade-off:** ratio 0.6 gives fewer points but a lower RMSE; lower detector thresholds
  (SIFT contrast 0.01, AKAZE 1e-4) give more points with slightly higher error.
- **ORB** is fastest but gives sparse clouds with ~2x the MAD of SIFT/AKAZE and low coverage.
  **AKAZE** matches SIFT's accuracy with fewer points and less time.
- **Error grows with range** (`q1_fig06.png`), as expected for the small 8 cm baseline
  (depth resolution ~ Z^2 / (f B), about 3 cm per pixel of disparity at 2 m).
- **Largest errors are on coral edges / overhangs** (`q1_fig04.png`), where the 2.5D reference grid cannot
  represent the structure and occlusions differ between the two views.
- **Types of structures** (`q1_fig10.png`, terrain split by slope): steep coral (> 37 deg) has ~2.5x the median
  error of flat sand/rubble (< 20 deg) for SIFT (1.8 vs 0.7 cm). SIFT + CLAHE covers ~99% of the imaged area in
  every slope class; ORB 10000 covers 76% of flat vs 68% of steep cells.
- **Outlier rejection stages** (printed): candidates 1.03M points, RMSE 1.55 m -> ratio test 472k, 0.23 m ->
  epipolar check 466k, 0.040 m -> reprojection check (removes 0) -> depth check 466,442, 0.038 m.

Figures in `figures_q1/`:

| File | Content |
|---|---|
| `q1_fig01.png` | SIFT keypoints, left and right (pair 20) |
| `q1_fig02.png` | SIFT inlier vs rejected matches (pair 20) |
| `q1_fig03.png` | 3D point cloud (true colour + camera path) + cloud over reference terrain |
| `q1_fig04.png` | Footprint on terrain, spatial height-error map, error histogram |
| `q1_fig05.png` | Detector / parameter comparison bar charts |
| `q1_fig06.png` | Median error vs distance from camera |
| `q1_fig07.png` | SIFT with no ratio test: inlier vs rejected matches |
| `q1_fig08.png` | ORB 10000: inlier vs rejected matches |
| `q1_fig09.png` | AKAZE default: inlier vs rejected matches |
| `q1_fig10.png` | Coverage and median error by terrain slope (flat / moderate / steep) |

---

## Question 2 - what is required

Data: `assignment2_places/assignment2_places/` - 7 classes (`ball_pit, desert, park, road, sky, snow, urban`),
50 RGB images each (256x256). Label = folder name. No other data may be used.

- `assign2_sceneclassifier.py` with `class_labels = assign2_sceneclassifier(image_path)` returning a list of class
  names (folder names), most likely first. Include the trained model weights file.
- At least a KNN or SVM classifier.
- A `mainQ2.py` script that runs all the code and produces all the plots.

Marking: code (10) works, readable, commented. Report (40, 4-page limit excl. references/appendix):
- Introduction (5) - scene classification, applications, algorithms used previously.
- Methodology (25) - pre-processing (colour space, quantisation, equalisation...) and feature extraction explained
  and justified; proper experimental methodology (train/test split, cross-validation) to choose the model; suitable
  metrics (accuracy, precision, recall, F1, top-k...); parameter selection justified.
- Discussion (10) - explain misclassifications and why features work; compare with state of the art.

---

## Question 2 - how to run

```bash
~/miniconda3/envs/CleanerS/bin/python MainQ2.py
```

Requirements: `numpy`, `opencv-python`, `scikit-learn` (model built with 1.0.2), `matplotlib`. Runtime ~5 min
(single process: joblib worker processes crash in the `CleanerS` env, so no `n_jobs`). Figures open as they are
drawn and are saved to `figures_q2/`; closing all windows ends the script. The run is deterministic
(`RANDOM_SEED = 0`).

Using the classifier on its own (needs `assign2_scene_model.pkl` next to it):

```python
from assign2_sceneclassifier import assign2_sceneclassifier
assign2_sceneclassifier('assignment2_places/assignment2_places/urban/00000001.jpg')
# ['urban', 'park', 'ball_pit', 'snow', 'sky', 'road', 'desert']
```

or `python assign2_sceneclassifier.py image1.jpg image2.jpg ...`.

Settings at the top of `MainQ2.py`:

| Variable | Default | Meaning |
|---|---|---|
| `dataset_dir` | `assignment2_places/assignment2_places` | Folder with one sub-folder per class |
| `TEST_FRACTION` | `0.2` | Held-out test set (stratified): 280 train / 70 test |
| `CV_FOLDS`, `CV_REPEATS` | `5`, `3` | Repeated stratified k-fold CV on the training set |
| `RANDOM_SEED` | `0` | Split, folds, random forest |
| `FINAL_TRAIN_ON_ALL` | `True` | Retrain the selected model on all 350 images before saving |
| `FEATURE_SETS`, `CLASSIFIERS`, `PREPROC_EXPERIMENTS` | - | What the experiments compare |

Feature settings (`FEATURE_PARAMS`) are at the top of `assign2_sceneclassifier.py`.

---

## Question 2 - method summary

The code follows the **Week 7 tutorial** structure: features -> scikit-learn classifier -> metrics.

1. **Split:** stratified 280 train / 70 test. Every choice below uses 3x5-fold CV on the training set only.
2. **Pre-processing:** resize to 128x128 colour + 128x128 grayscale. Colour space, colour quantisation (histogram
   bins), brightness equalisation of the colour image and CLAHE on the grayscale image are tested.
3. **Features (962 values, 5 blocks):** `colour` HSV 8x4x4 histogram (sqrt), `layout` Lab mean/std on a 4x4 grid,
   `hog` (32 px cells, `cv2.HOGDescriptor`), `lbp` rotation-invariant uniform LBP at r = 1, 2, 3, `gist` Gabor
   energy (4 scales x 6 orientations) on a 4x4 grid.
4. **Scaling:** `BlockScaler` - standardise, then give every block equal total variance.
5. **Classifiers:** KNN, SVM (linear / RBF) and random forest, each tuned by `GridSearchCV`.
6. **Selection:** highest CV accuracy -> **all features + RBF SVM (C = 10, gamma = 0.03)**.
7. **Evaluation:** once on the test set - accuracy, macro and per-class precision/recall/F1, top-k, confusion matrix.
8. **Deployment:** retrain on all 350 images, pickle to `assign2_scene_model.pkl`.

---

## Question 2 - results

Pre-processing / feature parameters (RBF SVM, 3x5-fold CV): layout grid 4x4 0.944 vs 2x2 0.873; HOG cell 32 0.627
vs 16 0.583; GIST grid 4x4 0.652 vs 2x2 0.636; Lab histogram 0.755 vs RGB 0.831 / HSV 0.824; texture with CLAHE
0.757 vs without 0.748 (RGB/HSV and CLAHE differences are within noise -> simpler option kept). Colour quantisation
(HSV bins): 4x2x2 0.751, 8x4x4 (used) 0.824, 16x4x4 0.827, 8x8x8 0.855. Histogram-equalising the colour image's
brightness lowers colour+layout from 0.933 to 0.912 (overall brightness is informative), so it is off.

Feature set x classifier (3x5-fold CV accuracy, best parameters):

| Feature set | KNN | SVM | Random forest |
|---|---|---|---|
| colour | 0.833 | 0.829 | 0.799 |
| layout | 0.906 | 0.944 | 0.935 |
| hog | 0.506 | 0.627 | 0.576 |
| lbp | 0.546 | 0.669 | 0.577 |
| gist | 0.592 | 0.652 | 0.648 |
| colour+layout | 0.904 | 0.933 | 0.949 |
| hog+lbp+gist | 0.710 | 0.751 | 0.711 |
| **all** | 0.914 | **0.963** | 0.960 |

Test set (70 images, 10 per class):

| Model | Accuracy | Macro F1 | Top-2 | Top-3 |
|---|---|---|---|---|
| KNN (all, k=5, distance) | 0.886 | 0.884 | 0.971 | 0.986 |
| **SVM (all, RBF) - selected** | **0.971** | **0.972** | **1.000** | **1.000** |
| Random forest (all) | 0.957 | 0.958 | 0.986 | 1.000 |

Discussion points for the report:
- **Colour layout is the strongest single cue** (0.944 alone): these classes have consistent spatial colour
  structure (sky on top, ground colour below). Texture alone is weak (0.63-0.75) but adds 3 points on top of colour.
- **SVM > random forest > KNN**; KNN is hurt by uninformative dimensions in the distance.
- **Parameters:** RBF SVM has a wide plateau (C >= 10, gamma 0.01-0.1), so the choice is robust; KNN best at k = 5.
- **Errors:** both test errors are predicted as snow (a sky with bare branches; a road under a white overcast sky);
  the true class is second in both cases, so top-2 = 100%.
- **Learning curve** still rising at 224 images -> more data would help; the saved model uses all 350.
- **Which classes each feature type separates** (`q2_fig07.png`, CV predictions on the training set): colour alone
  fails on the grey/white classes (urban recall 0.65, snow 0.72); layout fixes them (0.95); texture is the reverse
  of colour (urban 0.90, but desert 0.57, snow 0.53). With colour+layout the main confusions are road<->urban (6),
  sky<->urban (5) and sky<->snow (4); adding texture removes road<->urban entirely.
- **State of the art:** CNNs trained on Places365 reach far higher accuracy on hundreds of classes; hand-crafted
  features work here because there are only 7 visually distinct classes. Pre-trained CNNs were not used (the
  assignment forbids other data).

Figures in `figures_q2/`:

| File | Content |
|---|---|
| `q2_fig01.png` | Example images of each class |
| `q2_fig02.png` | Visualisation of each feature block |
| `q2_fig03.png` | Pre-processing / feature-parameter comparison |
| `q2_fig04.png` | Feature set x classifier CV accuracy |
| `q2_fig05.png` | Parameter selection (KNN k, RBF SVM C x gamma, linear SVM C) |
| `q2_fig06.png` | Learning curve of the selected model |
| `q2_fig07.png` | CV confusion matrix (training set) + per-class recall of each feature set |
| `q2_fig08.png` | Test confusion matrix + per-class precision/recall/F1 |
| `q2_fig09.png` | Top-k test accuracy |
| `q2_fig10.png` | Misclassified test images |

---

## AI use

Claude (Claude Code) was used to help write `MainQ1.py`, `MainQ2.py`, `assign2_sceneclassifier.py` and this documentation. This must be acknowledged in the
AI-use statement at the end of the report.
