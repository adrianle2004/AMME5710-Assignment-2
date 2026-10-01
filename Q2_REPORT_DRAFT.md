# Question 2 - Report Draft

Draft text for every part of the Question 2 report section, written from the results printed by `MainQ2.py`.
Rewrite it in your own words, check every reference yourself, and acknowledge the AI help in the AI-use statement.

**Limits from the spec:** Question 2 section = 4 pages, excluding references and appendices. The marking is
Introduction 5, **Methodology 25**, Discussion 10 (+ code 10). The report "must cover": pre-processing (colour space,
quantisation, histogram equalisation), feature extraction (why each method, intuition), algorithm selection (at least
KNN or SVM), model training (train/test split, cross-validation, suitable metrics) and parameter selection.

**Suggested page budget:** Introduction ~0.4 page, Methodology ~2.3 pages, Results and Discussion ~1.3 pages.
Methodology carries most marks, so most figures go there.

Text in *[square brackets and italics]* is a note to you, not report text.

---

## Question 2

### 2.1 Introduction *[5 marks: what scene classification is, applications, algorithms used previously]*

Scene classification assigns a semantic label to a whole image according to the type of place it shows (for example
"desert", "urban" or "snow"). Unlike object recognition, the label depends on the overall layout, surfaces and
materials of the scene rather than on a single object [1], [2]. It is used to give context to other vision tasks:
robots and vehicles use it for localisation and to adapt their behaviour to the environment, cameras use it to choose
exposure and colour settings, and photo collections and search engines use it to organise and retrieve images [1].

Early methods used global low-level features: colour histograms [3] and edge or texture statistics, for example to
separate indoor from outdoor images. Oliva and Torralba's GIST descriptor [4] summarised the "spatial envelope" of a
scene with Gabor filter energies on a coarse grid. Later methods used histograms of local features - bag of visual
words over SIFT descriptors [5] and spatial pyramid matching [6] - classified with SVMs or nearest neighbours. Since
large datasets such as Places [1] and SUN [7] became available, convolutional neural networks trained on millions of
images have become the state of the art.

This section builds a classifier for 7 classes of the Places dataset (ball pit, desert, park, road, sky, snow,
urban; 50 images each) using hand-crafted colour and texture features and classical classifiers, since no other data
may be used.

### 2.2 Methodology *[25 marks: pre-processing and features explained and justified; proper experimental methodology; suitable metrics]*

**Experimental design.** The 350 images are split once, stratified by class, into a training set of 280 images
(40 per class) and a test set of 70 (10 per class), with a fixed random seed. All choices - pre-processing, features,
classifier and parameters - are made with cross-validation on the training set only. The test set is used once, at
the end, to report the performance of the selected model. With only 56 images per validation fold, a single 5-fold
split is noisy (one image = 1.8% accuracy), so 5-fold cross-validation is repeated 3 times with different folds
(15 train/validation splits) and the mean and standard deviation are reported. Classes are balanced, so accuracy is
the selection metric. On the test set, per-class precision, recall and F1 (with their macro average), the confusion
matrix, and top-k accuracy are also reported. Top-k is useful because the classifier returns a ranked list of
labels, as the assignment allows.

**Pre-processing.** Each image is resized to 128x128 (area interpolation): colour statistics do not need full
resolution, and the classifier then also accepts images of other sizes. A colour copy is used for the colour
features and a grayscale copy for the texture features. Four pre-processing choices were tested with an RBF SVM
(Table 1, `q2_fig03.png`):
- *Colour space of the histogram:* HSV and RGB give the same accuracy (0.824 and 0.831, within one standard
  deviation); Lab is clearly worse (0.755). HSV was kept because it separates hue from brightness, so the same
  surface in sun and shade falls in the same hue bins.
- *Quantisation:* a joint HSV histogram with 4x2x2 bins is too coarse (0.751); 8x4x4 (used) gives 0.824, 16x4x4
  0.827 and 8x8x8 0.855. Finer saturation and value bins help the colour histogram on its own; 8x4x4 was kept
  for a smaller feature vector (128 vs 512 values). *[You could mention 8x8x8 as a possible improvement.]*
- *Histogram equalisation of the colour image:* equalising the brightness channel lowers the colour+layout accuracy
  from 0.933 to 0.912. The overall brightness of an image carries class information (bright snow and sky, dark ball
  pits and shaded streets), and equalisation removes it.
- *CLAHE on the grayscale image before the texture features:* no significant effect (0.757 vs 0.748, std ~0.05),
  so it is not used.

*Table 1 - Pre-processing and feature parameters (RBF SVM, 3x5-fold CV accuracy, mean +- std). Bold = used.*

| Variant | Accuracy | Variant | Accuracy |
|---|---|---|---|
| Colour hist RGB 4x4x4 | 0.831 +- 0.030 | Layout grid 2x2 | 0.873 +- 0.044 |
| **Colour hist HSV 8x4x4** | 0.824 +- 0.036 | **Layout grid 4x4** | 0.944 +- 0.027 |
| Colour hist Lab 4x4x4 | 0.755 +- 0.053 | HOG cell 16 px | 0.583 +- 0.059 |
| HSV 4x2x2 | 0.751 +- 0.045 | **HOG cell 32 px** | 0.627 +- 0.043 |
| HSV 16x4x4 | 0.827 +- 0.030 | GIST grid 2x2 | 0.636 +- 0.054 |
| HSV 8x8x8 | 0.855 +- 0.034 | **GIST grid 4x4** | 0.652 +- 0.059 |
| **Colour+layout** | 0.933 +- 0.028 | **Texture, no CLAHE** | 0.748 +- 0.036 |
| Colour+layout, brightness equalised | 0.912 +- 0.043 | Texture with CLAHE | 0.757 +- 0.053 |

**Feature extraction.** Each image is described by five complementary feature blocks (962 values;
`q2_fig02.png` shows what each one captures). The intuition is that these classes differ in *which colours* appear,
*where* they appear, and *what texture* the surfaces have:
1. *Colour histogram* (128 values): joint HSV histogram, normalised and square-rooted (Hellinger mapping) so that one
   dominant colour, such as a sky filling most of the image, does not swamp the other bins. Captures orange desert,
   green park, white snow, blue sky and the saturated multicolour ball pit.
2. *Spatial colour layout* (96): mean and standard deviation of L, a and b in each cell of a 4x4 grid. Lab is used
   because distances in Lab roughly follow perceived colour differences. Captures where colours are - sky at the top
   and snow or grass at the bottom - and whether a region is uniform or busy. A 4x4 grid is clearly better than 2x2
   (0.944 vs 0.873).
3. *HOG* (324): histograms of gradient orientation [8] with 32 px cells, describing the dominant edge directions in
   each region: vertical and horizontal building edges, converging road edges, the horizon. Large cells were better
   than 16 px (0.627 vs 0.583): fewer, more stable features for 280 training images.
4. *Local binary patterns* (30): rotation-invariant uniform LBP histograms [9] at radii 1, 2 and 3 px, describing
   fine texture: smooth sky, sand and snow against grass, foliage, balls and windows.
5. *GIST* (384): energy of 24 Gabor filters (4 scales x 6 orientations) averaged on a 4x4 grid [4], describing how
   much texture of each scale and direction lies in each region.

Blocks of very different sizes are combined with a block-wise scaler: each feature is standardised on the training
data and each block is divided by the square root of its size, so every block contributes equally to the distances
used by KNN and the SVM. The scaler is part of the classification pipeline, so it is fitted on the training folds
only.

**Algorithm selection.** Three classifiers were compared on eight feature sets (each block alone, colour+layout,
texture only, and all blocks): k-nearest neighbours [10], a support vector machine [11] with linear and RBF kernels,
and a random forest [12]. Each classifier's parameters were tuned by grid search with the same repeated
cross-validation (Table 2, `q2_fig04.png`).

*Table 2 - Feature set x classifier (3x5-fold CV accuracy, best parameters).*

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

Colour layout is the strongest single block (0.944), and the texture blocks alone reach only 0.63-0.75 (chance is
0.14). But texture still improves on colour for every classifier (SVM 0.933 -> 0.963 when added to colour+layout),
because it separates classes whose colours overlap. The SVM is best or equal-best on almost every feature set; KNN is
weakest, because every feature counts equally in its distance, including uninformative ones. The model with the
highest CV accuracy - **all five blocks with an RBF SVM** - was selected. The random forest (0.960) is within noise
of it.

**Parameter selection.** *[Figure = `q2_fig05.png`.]*
- *KNN:* k in {1, ..., 21}, uniform or distance weighting, Euclidean or Manhattan distance. Accuracy peaks at k = 5
  with distance weighting (0.914); k = 1 overfits to single neighbours and large k blurs the classes.
- *SVM:* linear kernel with C in {0.01, ..., 100}; RBF kernel with C in {0.1, ..., 1000} and gamma in {0.01, ..., 3}.
  The RBF SVM has a broad plateau of ~0.96 for C >= 10 and gamma 0.01-0.1, and collapses for gamma >= 1, where each
  training image becomes its own island (0.45 at gamma = 3). The chosen C = 10, gamma = 0.03 lies inside the
  plateau, so the result does not depend on a precise value. A linear SVM reaches the same accuracy (0.96 for
  C >= 1), so the combined features are already almost linearly separable.
- *Random forest:* 500 trees, maximum features sqrt(d) or 20% of d.

The learning curve of the selected model (`q2_fig06.png`) rises from 0.70 with 33 training images to 0.96 with 224
and is still rising, while training accuracy stays at 1.0. More data would still help, so the deployed model is
retrained on all 350 images after evaluation. The SVM outputs class probabilities (Platt scaling), and
`assign2_sceneclassifier` returns all seven labels ordered by probability.

### 2.3 Results and Discussion *[10 marks: explain misclassifications and why features work; compare with state of the art]*

**Test results.** *[Figures = `q2_fig08.png` (confusion matrix + per-class metrics) and `q2_fig09.png` (top-k).]*

*Table 3 - Test set (70 images, 10 per class). Each classifier uses its best feature set and parameters from CV.*

| Model | Accuracy | Macro F1 | Top-2 | Top-3 |
|---|---|---|---|---|
| KNN (all, k = 5, distance) | 0.886 | 0.884 | 0.971 | 0.986 |
| **SVM (all, RBF) - selected** | **0.971** | **0.972** | **1.000** | **1.000** |
| Random forest (all) | 0.957 | 0.958 | 0.986 | 1.000 |

The selected model classifies 68 of the 70 test images correctly (accuracy 0.971, macro F1 0.972), consistent with
its cross-validation estimate (0.963 +- 0.032). Ball pit, desert, park and urban are perfect; road and sky have
recall 0.9, and snow has precision 0.83 because both errors are predicted as snow. In both cases the true class is
the second choice, so top-2 accuracy is 100%.

**Misclassifications.** *[Figure = `q2_fig10.png`.]* The first error is a sky image with bare tree branches at the
side: the white clouds and the fine, high-contrast branch texture resemble a snowy scene with trees. The second is a
road through a flat desert under a white overcast sky: the large, bright, uniform upper half matches the layout of
snow scenes. Both errors come from bright, low-saturation regions, which is where the colour cues of sky, snow and
road overlap.

The test set is too small to show systematic confusions, so the cross-validated predictions on the training set
were also examined (Table 4, `q2_fig07.png`).

*Table 4 - Per-class recall by feature set (SVM, 5-fold cross-validated predictions on the 280 training images).*

| Class | colour | layout | hog+lbp+gist | colour+layout | all |
|---|---|---|---|---|---|
| ball_pit | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| desert | 1.00 | 1.00 | 0.57 | 1.00 | 1.00 |
| park | 0.90 | 0.93 | 0.82 | 0.95 | 0.95 |
| road | 0.78 | 0.93 | 0.57 | 0.88 | 0.93 |
| sky | 0.78 | 0.90 | 0.82 | 0.88 | 0.95 |
| snow | 0.72 | 0.95 | 0.53 | 0.93 | 0.93 |
| urban | 0.65 | 0.95 | 0.90 | 0.82 | 0.93 |

**Why the features work.** Table 4 shows that the feature types are complementary:
- The *global colour histogram* is enough for classes with a unique colour (ball pit, desert, park) but fails for
  the grey and white classes (urban 0.65, snow 0.72), which share colours.
- *Spatial layout* resolves most of these: in snow scenes the white is at the bottom, in sky scenes at the top;
  roads have grey in the bottom-centre, cities around the edges (urban 0.95, snow 0.95).
- *Texture* behaves the opposite way to colour. It recognises urban scenes well (0.90) from the straight edges and
  repeated windows of buildings, but not smooth scenes such as desert (0.57) and snow (0.53), which have little
  texture.

With colour+layout, the most frequent confusions are road <-> urban (6 images), sky <-> urban (5) and sky <-> snow
(4). Adding texture removes the road <-> urban confusion entirely. The remaining errors - sky <-> snow (3) and
park <-> road (3) - are scenes that really do share both colour layout and texture: bright sky above a white
or snowy foreground, and roads lined with grass and trees.

**Comparison with the state of the art.** Current scene classifiers are CNNs trained on millions of images, such as
the Places365 models [1]. They learn their features from data instead of using hand-designed ones and recognise
hundreds of categories, including indoor scenes and subtle differences (e.g. a forest road vs. a highway) that colour
and texture statistics cannot capture. *[Places365 CNNs reach roughly 55% top-1 / 85% top-5 accuracy on 365 classes
[1] - check the exact numbers in the paper before quoting them.]* On this small 7-class task, the hand-crafted
features reach 97% because the classes were chosen to be visually distinct and have strong, consistent colour layouts.
The approach would scale poorly to many classes, cluttered or indoor scenes, or classes that differ by their objects
rather than their surfaces. A CNN could not be trained from scratch on 280 images, and pre-trained networks were
not allowed because they use other data. Within these constraints, compact hand-crafted features with an SVM are an
appropriate choice: they need little data, train in seconds, and their errors can be explained.

**Limitations.** The test set has only 70 images (one image = 1.4%), so the difference between the SVM (0.971) and
the random forest (0.957) is not significant. The learning curve suggests that more training data would improve
accuracy further.

---

## Appendix B - Question 2 *[not counted in the 4 pages]*

Suggested content:
- **B.1 Additional figures:** `q2_fig01.png` (dataset examples), `q2_fig06.png` (learning curve) and
  `q2_fig09.png` (top-k) if they don't fit in the body.
- **B.2 Full grid-search results:** best parameters for every feature set x classifier (printed by `MainQ2.py`).
- **B.3 Feature settings:** `FEATURE_PARAMS` from `assign2_sceneclassifier.py`.
- **B.4 Code overview and usage:** `python MainQ2.py`; `assign2_sceneclassifier(image_path)` example; the model file
  `assign2_scene_model.pkl` must be next to `assign2_sceneclassifier.py`; built with scikit-learn 1.0.2.

---

## Figure plan for the body

| # | File | Caption (draft) |
|---|---|---|
| 1 | `q2_fig02.png` | What each feature block captures for an urban, a desert and a ball-pit image. |
| 2 | `q2_fig03.png` | Pre-processing and feature-parameter comparison (RBF SVM, 3x5-fold CV, mean +- std; blue = used). |
| 3 | `q2_fig04.png` | Cross-validation accuracy of every feature set with every classifier (best parameters). |
| 4 | `q2_fig05.png` | Parameter selection: KNN accuracy vs k (left), RBF SVM accuracy over C and gamma (centre), linear SVM vs C (right). |
| 5 | `q2_fig07.png` | Cross-validated confusion matrix of the selected model (left) and per-class recall by feature set (right). |
| 6 | `q2_fig08.png` | Test confusion matrix and per-class precision, recall and F1 of the selected model. |
| 7 | `q2_fig10.png` | The two misclassified test images with their top-2 predictions. |

*[Seven figures is a lot for 4 pages - combine or shrink them, or move 1, 2 or 4 to the appendix.]*

---

## References *[check every reference before using it]*

[1] B. Zhou, A. Lapedriza, A. Khosla, A. Oliva and A. Torralba, "Places: A 10 million image database for scene
recognition," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 40, no. 6, pp. 1452-1464, 2018
(online 2017). *[The assignment cites it as 2017.]*

[2] A. Torralba, K. P. Murphy, W. T. Freeman and M. A. Rubin, "Context-based vision system for place and object
recognition," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, 2003.

[3] M. J. Swain and D. H. Ballard, "Color indexing," *International Journal of Computer Vision*, vol. 7, no. 1,
pp. 11-32, 1991.

[4] A. Oliva and A. Torralba, "Modeling the shape of the scene: A holistic representation of the spatial envelope,"
*International Journal of Computer Vision*, vol. 42, no. 3, pp. 145-175, 2001.

[5] L. Fei-Fei and P. Perona, "A Bayesian hierarchical model for learning natural scene categories," in *Proc. IEEE
Conference on Computer Vision and Pattern Recognition (CVPR)*, 2005.

[6] S. Lazebnik, C. Schmid and J. Ponce, "Beyond bags of features: Spatial pyramid matching for recognizing natural
scene categories," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2006.

[7] J. Xiao, J. Hays, K. A. Ehinger, A. Oliva and A. Torralba, "SUN database: Large-scale scene recognition from abbey
to zoo," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2010.

[8] N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," in *Proc. IEEE Conference on
Computer Vision and Pattern Recognition (CVPR)*, 2005.

[9] T. Ojala, M. Pietikäinen and T. Mäenpää, "Multiresolution gray-scale and rotation invariant texture
classification with local binary patterns," *IEEE Transactions on Pattern Analysis and Machine Intelligence*,
vol. 24, no. 7, pp. 971-987, 2002.

[10] T. Cover and P. Hart, "Nearest neighbor pattern classification," *IEEE Transactions on Information Theory*,
vol. 13, no. 1, pp. 21-27, 1967.

[11] C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, pp. 273-297, 1995.

[12] L. Breiman, "Random forests," *Machine Learning*, vol. 45, pp. 5-32, 2001.

---

## Still to do for Question 2

| Item | Status |
|---|---|
| `assign2_sceneclassifier.py` with `assign2_sceneclassifier(image_path)`, returns ranked folder names | Done |
| Trained model file next to it (`assign2_scene_model.pkl`) | Done (scikit-learn 1.0.2 - the tutor's version should match, or re-run `MainQ2.py`) |
| KNN or SVM | Done (KNN, SVM, random forest) |
| Train/test split, cross-validation, metrics, parameter selection | Done |
| Pre-processing: colour space, quantisation, histogram equalisation | Done - printed by `MainQ2.py` + `q2_fig03.png` |
| Misclassification analysis | Done - test errors (`q2_fig10.png`) + CV per-class analysis (`q2_fig07.png`) |
| File name | The general rules ask for `mainQN.py`; currently `MainQ2.py` (same choice as Q1). |
| Optional | 8x8x8 HSV bins scored better on their own (0.855 vs 0.824); not tested in the full model. |
| Write the report | Use this draft; fit Q2 into 4 pages. |
| Appendix B (Q2) | Extra figures + settings + code usage (see above). |
| Code zip | Include `MainQ2.py`, `assign2_sceneclassifier.py`, `assign2_scene_model.pkl` (images not needed). |
