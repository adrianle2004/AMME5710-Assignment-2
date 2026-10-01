# Question 2 - How the Code Works

This document explains what the Question 2 code does: how each image is turned into a feature vector, how the
classifier and its parameters are chosen, how the model is evaluated, and how `assign2_sceneclassifier` produces
its ranked labels. The function reference is at the end. See `README.md` for how to run it.

The code is split in two files:
- [`assign2_sceneclassifier.py`](assign2_sceneclassifier.py) - feature extraction and the required
  `assign2_sceneclassifier(image_path)` function. Training and prediction both use these functions, so the
  features are always computed the same way.
- [`MainQ2.py`](MainQ2.py) - all experiments: data split, pre-processing and feature comparison, classifier and
  parameter selection, test-set evaluation, figures, and saving the final model to `assign2_scene_model.pkl`.

---

## 1. Data and split

**Code:** [`load_dataset`](MainQ2.py#L118), [split in `main`](MainQ2.py#L511)

| Data | Content |
|---|---|
| `assignment2_places/assignment2_places/<class>/` | 7 classes x 50 RGB images, all 256x256 |
| Classes | `ball_pit, desert, park, road, sky, snow, urban` (label = folder name) |

The 350 images are split once, **stratified** by class, into a **training set of 280** (40 per class) and a
**test set of 70** (10 per class), with a fixed seed (`RANDOM_SEED = 0`). All choices - pre-processing, features,
classifier, parameters - are made using only the training set, with **repeated stratified 5-fold cross-validation**
(5 folds x 3 repeats = 15 train/validation splits, `RepeatedStratifiedKFold`). The test set is used once, at the
end, to report the performance of the chosen model.

Repeating the 5-fold CV matters here: with only 56 images per validation fold, one image is 1.8% accuracy, and a
single 5-fold run changed which setting "won" when the same experiment was re-run with other settings changed.
Averaging 15 folds makes the means more stable; the std over the 15 folds is reported with every score.

## 2. Pre-processing

**Code:** [`preprocess`](assign2_sceneclassifier.py#L77), [`FEATURE_PARAMS`](assign2_sceneclassifier.py#L50)

Every image is resized (`cv2.INTER_AREA`) into two inputs:
- a **128x128 colour image** (BGR) for the colour features. Colour statistics do not need full resolution, and
  resizing also makes the function work on images that are not 256x256;
- a **128x128 grayscale image** for the texture features (HOG, LBP, Gabor).

Options tested (section 5): the colour space and the quantisation (number of bins) of the colour histogram,
histogram equalisation of the colour image's brightness (`colour_equalise`: `cv2.equalizeHist` on the V channel of
HSV, so hues are unchanged), and CLAHE on the grayscale image. Both equalisations are off in the final model.

## 3. Feature extraction

**Code:** [`extract_features`](assign2_sceneclassifier.py#L230), [`stack_blocks`](assign2_sceneclassifier.py#L236)

Each image is described by five **feature blocks**, 962 values in total. `plot_feature_examples`
(`q2_fig02.png`) shows what each block sees for an urban, a desert and a ball-pit image.

| Block | Size | What it measures | Why it helps for these classes |
|---|---|---|---|
| `colour` | 128 | Joint HSV histogram, 8 hue x 4 saturation x 4 value bins, normalised, square-rooted | Global colour: orange desert, green park, white snow, blue sky, saturated multicolour ball pit |
| `layout` | 96 | Mean and std of L, a, b in each cell of a 4x4 grid | *Where* the colours are: blue on top + white below = snow, green below = park, grey bottom-centre = road |
| `hog` | 324 | HOG, 32 px cells, 2x2-cell blocks, 9 orientation bins (`cv2.HOGDescriptor`, Week 7) | Dominant edge directions per region: building verticals/horizontals, converging road edges, horizon |
| `lbp` | 30 | Rotation-invariant uniform LBP histograms at radii 1, 2, 3 | Fine texture: smooth sky/sand/snow vs. grass, foliage, balls, windows |
| `gist` | 384 | Gabor energy, 4 wavelengths (4-32 px) x 6 orientations, averaged on a 4x4 grid | "Spatial envelope" (Oliva & Torralba): how much texture of each scale and direction is in each region |

### 3.1 Colour histogram

**Code:** [`colour_histogram`](assign2_sceneclassifier.py#L115)
`cv2.calcHist` over the three HSV channels gives a joint 8x4x4 histogram, divided by the pixel count. HSV separates
colour (hue) from brightness (value), so the same sand colour in shade and in sun falls into the same hue bins.
The square root of each bin is taken (Hellinger mapping): otherwise one dominant colour - a sky covering most of
the image - makes every other bin close to zero and the distance between images depends almost only on that bin.

### 3.2 Spatial colour layout

**Code:** [`spatial_layout`](assign2_sceneclassifier.py#L126), [`grid_mean`](assign2_sceneclassifier.py#L99)
The image is converted to Lab (distances roughly match perceived colour differences) and divided into a 4x4 grid.
For each cell the mean and standard deviation of L, a and b are stored (16 cells x 6 = 96 values, scaled 0-1).
The mean describes the colour of each region, the std whether the region is uniform (sky, sand) or busy (balls,
foliage). This is the single strongest block (section 6): these scene classes have very consistent layouts.

### 3.3 HOG

**Code:** [`hog_features`](assign2_sceneclassifier.py#L138)
Same `cv2.HOGDescriptor` as the Week 7 tutorial, with the window equal to the whole 128x128 image. With 32 px cells
the image has 4x4 cells and 3x3 overlapping blocks of 2x2 cells, each with a 9-bin gradient-orientation histogram
(3 x 3 x 4 x 9 = 324 values). Large cells describe the layout of the main edges rather than small details.

### 3.4 Local binary patterns

**Code:** [`lbp_riu2`](assign2_sceneclassifier.py#L149), [`lbp_histogram`](assign2_sceneclassifier.py#L170)
For every pixel, 8 neighbours at distance `r` are compared with the centre (1 if brighter or equal). If the circular
bit pattern has at most 2 transitions (a "uniform" pattern: flat area, edge, corner, spot) its code is the number of
1s (0-8); otherwise it gets code 9. This code does not change when the image is rotated. The 10-bin histogram of
codes is computed at radii 1, 2 and 3 px (30 values). Implemented in numpy (scikit-image is not required).

### 3.5 GIST (Gabor filter bank)

**Code:** [`gabor_bank`](assign2_sceneclassifier.py#L184), [`gabor_energy_maps`](assign2_sceneclassifier.py#L202), [`gist_features`](assign2_sceneclassifier.py#L213)
24 Gabor filters (wavelengths 4, 8, 16, 32 px x 6 orientations) are applied to the grayscale image. Each filter is an
even/odd (cosine/sine) pair, and the magnitude `sqrt(even^2 + odd^2)` is the local energy at that scale and
orientation. The energy maps are averaged over a 4x4 grid (24 x 16 = 384 values).

### 3.6 Combining blocks: `BlockScaler`

**Code:** [`BlockScaler`](assign2_sceneclassifier.py#L246), [`make_pipeline`](MainQ2.py#L142)
When blocks are concatenated, each feature is standardised (zero mean, unit variance over the training data) and
then each block is divided by `sqrt(block size)`. Every block then has the same total variance, so in the KNN and
SVM distances a 384-value GIST block does not count 13x more than the 30-value LBP block. The scaler is the first
step of an sklearn `Pipeline`, so in cross-validation it is fitted on the training folds only.

## 4. Classifiers and parameter grids

**Code:** [`CLASSIFIERS`](MainQ2.py#L74), [`grid_search`](MainQ2.py#L154), [`prefix_grid`](MainQ2.py#L147)

Three classifiers (sklearn), each tuned by `GridSearchCV` (accuracy, 3x5-fold CV):

| Classifier | Parameters searched |
|---|---|
| KNN (`KNeighborsClassifier`) | k in {1, 3, 5, 7, 9, 11, 15, 21}; weights uniform / distance; metric euclidean / manhattan |
| SVM (`SVC`) | linear kernel, C in {0.01 ... 100}; RBF kernel, C in {0.1 ... 1000} x gamma in {0.01 ... 3} |
| Random forest | 500 trees; max_features sqrt / 0.2 |

With `BlockScaler`, squared distances between images are about 2 x (number of blocks), so the RBF gamma grid
0.01-3 spans kernels from very wide to very narrow for every feature set.

## 5. Pre-processing and feature-parameter experiment

**Code:** [`PREPROC_EXPERIMENTS`](MainQ2.py#L88), [loop in `main`](MainQ2.py#L530), [`plot_preproc_results`](MainQ2.py#L276)

Each variant re-extracts only the affected blocks and is scored with an RBF SVM (small grid, 3x5-fold CV):

| Variant | CV accuracy | | Variant | CV accuracy |
|---|---|---|---|---|
| colour hist RGB (4x4x4) | 0.831 +- 0.030 | | texture **no CLAHE** | 0.748 +- 0.036 |
| **colour hist HSV (8x4x4)** | 0.824 +- 0.036 | | texture with CLAHE | 0.757 +- 0.053 |
| colour hist Lab (4x4x4) | 0.755 +- 0.053 | | HOG cell 16 | 0.583 +- 0.059 |
| HSV bins 4x2x2 | 0.751 +- 0.045 | | **HOG cell 32** | 0.627 +- 0.043 |
| HSV bins 16x4x4 | 0.827 +- 0.030 | | GIST grid 2x2 | 0.636 +- 0.054 |
| HSV bins 8x8x8 | 0.855 +- 0.034 | | **GIST grid 4x4** | 0.652 +- 0.059 |
| **colour+layout** | 0.933 +- 0.028 | | layout grid 2x2 | 0.873 +- 0.044 |
| colour+layout, V equalised | 0.912 +- 0.043 | | **layout grid 4x4** | 0.944 +- 0.027 |

(bold = setting used; `q2_fig03.png`)

- **Clear effects:** a 4x4 layout grid beats 2x2 by 7 points (where things are matters); Lab histograms are
  clearly worse than RGB/HSV (a and b do not separate hues as well as H); HOG is better with 32 px cells than 16
  px (fewer, more stable features for 280 training images).
- **Quantisation (HSV bins):** 4x2x2 bins is too coarse (0.751). The colour histogram improves with finer bins:
  8x4x4 (used) 0.824, 16x4x4 0.827, 8x8x8 0.855. 8x8x8 scores best on its own, by about one standard deviation,
  but a separate check with all five blocks + SVM (same 15 folds) gave 0.961 vs 0.963 for 8x4x4: no gain once
  layout and texture are included. The final model keeps 8x4x4 (128 values instead of 512).
- **Equalising the colour image hurts** (colour+layout 0.933 -> 0.912): overall brightness is informative here
  (bright snow and sky, dark ball pits and shaded streets), and equalisation removes it.
- **No real effect:** RGB vs HSV and CLAHE vs no CLAHE differ by less than 1 point, far less than the std, and their
  order changed between runs. For these the simpler / more interpretable option is kept: HSV (hue is separate from
  brightness) and no CLAHE (one fewer processing step; equalising removes the brightness differences - bright snow
  and sky, dark ball-pit backgrounds - that are part of the scene).

## 6. Feature set and classifier comparison

**Code:** [`FEATURE_SETS`](MainQ2.py#L60), [loop in `main`](MainQ2.py#L548), [`plot_model_comparison`](MainQ2.py#L296)

Every feature set is tried with every classifier; the table shows the best CV accuracy of each grid search
(`q2_fig04.png`):

| Feature set | KNN | SVM | Random forest |
|---|---|---|---|
| colour | 0.833 | 0.829 | 0.799 |
| layout | 0.906 | 0.944 | 0.935 |
| hog | 0.506 | 0.627 | 0.576 |
| lbp | 0.546 | 0.669 | 0.577 |
| gist | 0.592 | 0.652 | 0.648 |
| colour+layout | 0.904 | 0.933 | 0.949 |
| hog+lbp+gist | 0.710 | 0.751 | 0.711 |
| **all** | 0.914 | **0.963 +- 0.032** | 0.960 |

- **Colour, and above all colour layout, carry most of the information** for these 7 classes; the texture blocks
  alone reach only 0.5-0.75 (chance = 0.14).
- **Texture still adds to colour:** all five blocks beat colour+layout for every classifier (SVM 0.933 -> 0.963),
  because texture separates the classes whose colours overlap (grey urban vs. grey road, white snow vs. white
  clouds).
- **SVM >= random forest > KNN** on almost every feature set. KNN suffers most from uninformative dimensions: every
  feature counts equally in its distance.

**Model selection** ([`main`](MainQ2.py#L561)): the combination with the highest mean CV accuracy is chosen -
**all features + RBF SVM, C = 10, gamma = 0.03 (CV 0.963 +- 0.032)**. The random forest (0.960) is within noise of
it; the SVM is also the classifier the assignment asks for.

## 7. Parameter selection

**Code:** [`plot_parameter_selection`](MainQ2.py#L318), [`plot_learning_curve`](MainQ2.py#L373)

`q2_fig05.png` shows the grid-search results for the chosen feature set (all blocks):
- **KNN:** accuracy peaks at k = 5 (0.914, distance weighting); k = 1 overfits to single neighbours (0.88), large
  k blurs classes together. Distance weighting is slightly better than uniform for every k.
- **RBF SVM:** a broad plateau at 0.96 for C >= 10 and gamma 0.01-0.1. Larger gamma (>= 1) makes each training image
  its own island and accuracy collapses (0.45 at gamma = 3). The chosen (C = 10, gamma = 0.03) is inside the
  plateau, so the result does not depend on a precise value.
- **Linear SVM:** 0.96 for C >= 1, the same as the best RBF within noise - the combined features are already almost
  linearly separable.

`q2_fig06.png` (learning curve of the chosen model): CV accuracy rises from 0.70 with 33 training images to 0.96
with 224 and is still rising slowly, while training accuracy stays at 1.0. More data would still help a little;
this is why the saved model is retrained on all 350 images (section 9).

## 8. Which classes each feature type separates

**Code:** [`CLASS_ANALYSIS_SETS`](MainQ2.py#L109), [`cv_class_analysis`](MainQ2.py#L194), [`print_class_analysis`](MainQ2.py#L215), [`plot_class_analysis`](MainQ2.py#L436)

The test set has only 70 images (2 errors), too few to see systematic confusions. So, for the selected classifier
type (SVM), the best model of each feature set predicts every **training** image once by 5-fold cross-validation
(`cross_val_predict`). This gives a confusion matrix over 280 images and the recall of each class for each feature
type (`q2_fig07.png`):

| Class | colour | layout | hog+lbp+gist | colour+layout | all |
|---|---|---|---|---|---|
| ball_pit | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| desert | 1.00 | 1.00 | 0.57 | 1.00 | 1.00 |
| park | 0.90 | 0.93 | 0.82 | 0.95 | 0.95 |
| road | 0.78 | 0.93 | 0.57 | 0.88 | 0.93 |
| sky | 0.78 | 0.90 | 0.82 | 0.88 | 0.95 |
| snow | 0.72 | 0.95 | 0.53 | 0.93 | 0.93 |
| urban | 0.65 | 0.95 | 0.90 | 0.82 | 0.93 |
| accuracy | 0.832 | 0.950 | 0.746 | 0.921 | 0.954 |

- **Colour alone** separates the classes with a unique colour (ball pit, desert, park) but not the grey/white
  classes: urban 0.65, snow 0.72, road and sky 0.78.
- **Layout fixes these** (urban 0.95, snow 0.95): white at the bottom vs. white at the top, grey road in the
  bottom-centre vs. grey buildings around the edges.
- **Texture** is the reverse of colour: good for urban (0.90, building edges and windows) and ball pit, poor for the
  smooth classes (desert 0.57, snow 0.53).
- **Most-confused pairs:** with colour+layout, road <-> urban (6 images), sky <-> urban (5) and sky <-> snow (4).
  With all features, road <-> urban drops to 0 and the remaining errors are sky <-> snow (3), park <-> road (3) and
  sky <-> urban (2). Adding texture removes most of the urban confusions, which is why it improves on colour+layout.

## 9. Test-set evaluation

**Code:** [`with_probabilities`](MainQ2.py#L162), [`evaluate`](MainQ2.py#L171), [test loop in `main`](MainQ2.py#L578), [`plot_test_results`](MainQ2.py#L391), [`plot_topk`](MainQ2.py#L421), [`plot_misclassified`](MainQ2.py#L468)

The best configuration of each classifier type is trained on the 280 training images and scored once on the 70 test
images. The SVM is given `probability=True` (Platt scaling) so that it can rank all classes; the predicted class is
the first of this ranking - exactly what `assign2_sceneclassifier` returns.

| Model (best CV config) | Accuracy | Macro F1 | Top-2 | Top-3 |
|---|---|---|---|---|
| KNN (all, k=5, distance) | 0.886 | 0.884 | 0.971 | 0.986 |
| **SVM (all, RBF) - selected** | **0.971** | **0.972** | **1.000** | **1.000** |
| Random forest (all) | 0.957 | 0.958 | 0.986 | 1.000 |

Metrics used: **accuracy** (the classes are balanced, so it is not misleading), **per-class precision / recall /
F1** and their macro average (shows which classes are confused), **top-k accuracy** (the function returns a ranked
list), and the **confusion matrix**.

Selected model, per class (`q2_fig08.png`): ball_pit, desert, park and urban are perfect; road and sky have recall
0.9; snow has precision 0.83 because both errors are predicted as snow. `q2_fig10.png` shows the two errors:
- a **sky** image with bare tree branches on the right - the white clouds and the branch texture look like a snowy
  scene;
- a **road** through a flat desert under a white overcast sky - the large bright, uniform upper half matches snow.

In both cases the true class is the **second** choice, hence top-2 accuracy = 1.0. The test accuracy (0.971) agrees
with the CV estimate (0.963 +- 0.032); with 70 test images, one image is 1.4%, so the two are consistent.

## 10. Final model and `assign2_sceneclassifier`

**Code:** [final model in `main`](MainQ2.py#L603), [`assign2_sceneclassifier`](assign2_sceneclassifier.py#L292), [`load_model`](assign2_sceneclassifier.py#L281), [`rank_classes`](assign2_sceneclassifier.py#L270)

After evaluation, the selected pipeline (BlockScaler + SVM with the chosen parameters) is retrained on **all 350
images** (`FINAL_TRAIN_ON_ALL = True`) and saved with `pickle` to `assign2_scene_model.pkl` (~2 MB), together with
the feature blocks and the `FEATURE_PARAMS` used. The reported test metrics are from the model trained on the 280
training images only.

`assign2_sceneclassifier(image_path)`:
1. loads the model file once (from the folder of `assign2_sceneclassifier.py`, so it works from any working
   directory);
2. reads the image with `cv2.imread` (3-channel, also for grayscale files) and extracts the same feature blocks with
   the saved settings;
3. returns **all 7 labels ordered by predicted probability**, e.g.
   `['urban', 'park', 'ball_pit', 'snow', 'sky', 'road', 'desert']`; element 0 is the predicted scene.

The pickled model needs the same scikit-learn version to load reliably (built with scikit-learn 1.0.2). If the
versions differ, re-running `MainQ2.py` rebuilds it.

## 11. Figures

| Figure | Content |
|---|---|
| `q2_fig01.png` | 5 example images of each class |
| `q2_fig02.png` | What each feature block sees: HSV histogram, 4x4 layout colours, LBP codes, Gabor energy (urban, desert, ball pit) |
| `q2_fig03.png` | Pre-processing / feature-parameter CV accuracy (blue = setting used) |
| `q2_fig04.png` | CV accuracy of every feature set x classifier |
| `q2_fig05.png` | Parameter selection: KNN accuracy vs k, RBF SVM C x gamma heat map, linear SVM accuracy vs C |
| `q2_fig06.png` | Learning curve of the selected model |
| `q2_fig07.png` | Cross-validated confusion matrix (all features + SVM, training set) and per-class recall of each feature set |
| `q2_fig08.png` | Test confusion matrix and per-class precision / recall / F1 of the selected model |
| `q2_fig09.png` | Top-k test accuracy of the best KNN, SVM and random forest |
| `q2_fig10.png` | The misclassified test images with their true label and top-2 predictions |

Figures are shown while the script runs and saved to `figures_q2/` as soon as they are drawn (same mechanism as Q1,
[`show_figures`](MainQ2.py#L488)).

---

## 12. Function reference

### Settings
- `dataset_dir` (`MainQ2.py`) - dataset folder with one sub-folder per class.
- `TEST_FRACTION`, `CV_FOLDS`, `CV_REPEATS`, `RANDOM_SEED` - test split and cross-validation.
- `FINAL_TRAIN_ON_ALL` - retrain the selected model on all images before saving.
- `FEATURE_SETS`, `CLASSIFIERS`, `PREPROC_EXPERIMENTS`, `CLASS_ANALYSIS_SETS` - what the experiments compare.
- `FEATURE_PARAMS`, `BLOCKS` (`assign2_sceneclassifier.py`) - feature settings and block names.

### `assign2_sceneclassifier.py`

| Function | What it does |
|---|---|
| [`preprocess(img_bgr, params)`](assign2_sceneclassifier.py#L77) | Resizes the image to the colour and grayscale inputs; optional brightness equalisation of the colour image and CLAHE on the grayscale image. |
| [`grid_mean(values, n)`](assign2_sceneclassifier.py#L99) | Averages an image (or stack of maps) over an n x n grid. |
| [`colour_histogram(colour, params)`](assign2_sceneclassifier.py#L115) | Square-rooted, normalised joint colour histogram. |
| [`spatial_layout(colour, params)`](assign2_sceneclassifier.py#L126) | Mean and std of Lab in each grid cell. |
| [`hog_features(gray, params)`](assign2_sceneclassifier.py#L138) | HOG descriptor of the whole image. |
| [`lbp_riu2(gray, radius)`](assign2_sceneclassifier.py#L149) | Rotation-invariant uniform LBP code of every pixel. |
| [`lbp_histogram(gray, params)`](assign2_sceneclassifier.py#L170) | Normalised LBP code histograms at several radii. |
| [`gabor_bank(wavelengths, n_orient)`](assign2_sceneclassifier.py#L184) | Builds (and caches) the even/odd Gabor kernel pairs. |
| [`gabor_energy_maps(gray, params)`](assign2_sceneclassifier.py#L202) | Gabor energy map for each filter. |
| [`gist_features(gray, params)`](assign2_sceneclassifier.py#L213) | Gabor energies averaged on a grid (GIST). |
| [`extract_features(img_bgr, params, blocks)`](assign2_sceneclassifier.py#L230) | All requested feature blocks of one image. |
| [`stack_blocks(features, blocks)`](assign2_sceneclassifier.py#L236) | Concatenates blocks; returns the matrix and block sizes. |
| [`BlockScaler`](assign2_sceneclassifier.py#L246) | Standardises features and gives every block equal total variance. |
| [`rank_classes(model, X)`](assign2_sceneclassifier.py#L270) | Class labels ordered by predicted probability. |
| [`load_model(path)`](assign2_sceneclassifier.py#L281) | Loads the saved model file once. |
| [`assign2_sceneclassifier(image_path)`](assign2_sceneclassifier.py#L292) | Classifies one image file; returns all labels, most likely first. |

### `MainQ2.py`

| Function | What it does |
|---|---|
| [`load_dataset(root)`](MainQ2.py#L118) | Reads all images; label = folder name. |
| [`extract_all(images, params, blocks)`](MainQ2.py#L133) | Feature blocks of every image as (N, d) arrays. |
| [`make_pipeline(block_sizes, clf)`](MainQ2.py#L142) | `BlockScaler` + classifier pipeline. |
| [`prefix_grid(grid)`](MainQ2.py#L147) | Adds the pipeline step name to grid parameter names. |
| [`grid_search(X, y, block_sizes, clf, grid, cv)`](MainQ2.py#L154) | Cross-validated grid search (accuracy). |
| [`with_probabilities(model)`](MainQ2.py#L162) | Unfitted copy of a pipeline whose SVM outputs probabilities. |
| [`evaluate(model, X, y, classes)`](MainQ2.py#L171) | Accuracy, per-class precision/recall/F1, top-k, confusion matrix. |
| [`cv_class_analysis(...)`](MainQ2.py#L194) | Cross-validated predictions on the training set for several feature sets: per-class recall, confusion matrices, most-confused pairs. |
| [`print_class_analysis(...)`](MainQ2.py#L215) | Prints the per-class recall table and most-confused pairs. |
| [`describe_params(params)`](MainQ2.py#L189) | Text of the best grid-search parameters. |
| [`rgb(img_bgr)`](MainQ2.py#L229) | BGR -> RGB for plotting. |
| [`plot_dataset_examples(...)`](MainQ2.py#L233) | Example images of each class. |
| [`plot_feature_examples(...)`](MainQ2.py#L247) | Visualises each feature block for example images. |
| [`plot_preproc_results(results)`](MainQ2.py#L276) | Bar chart of the pre-processing / feature-parameter experiment. |
| [`plot_model_comparison(table)`](MainQ2.py#L296) | Grouped bars: feature set x classifier CV accuracy. |
| [`plot_parameter_selection(...)`](MainQ2.py#L318) | KNN k curve, RBF SVM heat map, linear SVM C curve. |
| [`plot_learning_curve(...)`](MainQ2.py#L373) | CV accuracy vs number of training images. |
| [`plot_test_results(m, classes, title)`](MainQ2.py#L391) | Confusion matrix and per-class metric bars. |
| [`plot_topk(test_results, n_classes)`](MainQ2.py#L421) | Top-k test accuracy curves. |
| [`plot_class_analysis(...)`](MainQ2.py#L436) | CV confusion matrix and per-class recall bars (`q2_fig07.png`). |
| [`plot_misclassified(...)`](MainQ2.py#L468) | Grid of the misclassified test images. |
| [`show_figures(saved)`](MainQ2.py#L488) | Saves new figures to `figures_q2/` and draws them. |
| [`main()`](MainQ2.py#L499) | Runs all experiments, evaluates, saves the model and figures. |
