# Question 1 - Report Draft

Draft text for every part of the Question 1 report section, written from the results of `MainQ1.py`.
Rewrite it in your own words, check every reference yourself, and acknowledge the AI help in the AI-use statement.

**Limits from the spec:** Question 1 section = 4 pages, not counting the appendix. Subsections must be
*Introduction*, *Methodology*, *Results and Discussion*. The appendix needs a Question 1 subsection.

**Suggested page budget:** Introduction ~0.5 page, Methodology ~1 page, Results and Discussion ~2.5 pages (most marks:
15/30). Use 4-6 figures in the body and move the rest to the appendix.

Text in *[square brackets and italics]* is a note to you, not report text.

---

## Question 1

### 1.1 Introduction *[5 marks: how stereo vision works, applications, algorithms]*

Stereo vision recovers the 3D structure of a scene from two images taken at the same time by cameras a known
distance apart (the baseline). A scene point projects to slightly different positions in the two images; this
shift is the disparity. Once the cameras are calibrated - their intrinsic parameters (focal length, principal point,
lens distortion) and the rotation and translation between them are known - each pixel in one image can only match
pixels along a line in the other image, the epipolar line [1]. Finding the matching point (the correspondence
problem) and intersecting the two viewing rays (triangulation) gives the point's 3D position. For a rectified pair,
depth is Z = fB/d, where f is the focal length, B the baseline and d the disparity, so depth resolution falls with
the square of the distance [1].

Correspondences can be found densely, for every pixel, or sparsely, at distinctive interest points. Dense methods
compare image windows along epipolar lines and regularise the result with local, global or semi-global
optimisation [2], [3]; semi-global matching [3] is widely used in robotics and cars. Sparse methods detect interest
points and match their descriptors: SIFT [4] uses scale-space extrema and gradient histograms, ORB [5] uses fast
corner detection and binary descriptors, and AKAZE [6] uses a non-linear scale space with binary descriptors. Wrong
matches are removed with descriptor tests such as the ratio test [4] and with the epipolar constraint, often
enforced with RANSAC [7].

Stereo vision is used wherever metric 3D structure is needed from passive sensors: obstacle detection and visual
odometry on planetary rovers [8], depth perception for autonomous cars, benchmarked by datasets such as KITTI [9],
and mapping of the seafloor from underwater vehicles and diver rigs [10], [11]. Underwater stereo is especially
useful because acoustic sensors have much lower resolution at close range. It is also harder: light is absorbed and
scattered, so images are dark, low-contrast and colour-shifted, and the housing adds lens distortion [10].
Underwater 3D mapping of coral reefs is used to measure reef structure and monitor change over time [10], [11].

This section reconstructs a coral reef from 49 stereo pairs captured by a diver-operated rig with a colour and a
monochrome camera, using sparse feature matching, and evaluates the result against a reference terrain model.

### 1.2 Methodology *[10 marks: features extracted and matched; world point cloud built correctly]*

**Data and coordinate frames.** The stereo calibration gives the intrinsic matrices K_L, K_R, distortion
coefficients and the pose (R, t) of the right camera relative to the left. The baseline is 8.0 cm, along the image
y-axis, and f = 1723 px. For each frame i, the pose data give (R_i, t_i), which map world coordinates into the left
camera frame: x_cam = R_i X_world + t_i. With this convention the camera centres, C_i = -R_i^T t_i, lie at
Z = 2.4-3.0 m, about 1.2-3 m above a seafloor at Z = 3.7-5.7 m, with the optical axes pointing down, which confirms
it. In this world frame Z points
down and the reference surface is Z = -height_grid.

**Pipeline.** Each stereo pair is processed as follows *[Figure 1 would be a good place for a block diagram]*:

1. *Pre-processing.* The left colour image is converted to grayscale. Contrast-limited adaptive histogram
   equalisation (CLAHE, clip limit 2, 8x8 tiles) [12] is applied to both images, to raise the low underwater
   contrast and make the colour and monochrome sensors look more alike.
2. *Feature extraction.* SIFT keypoints and 128-D descriptors are extracted from both images (~21,000 per image).
3. *Matching.* Each left descriptor is matched to its two nearest right descriptors by brute-force L2 search. A match
   is kept only if the nearest distance is below 0.8 of the second-nearest (Lowe's ratio test [4]), which removes
   ambiguous matches on repetitive texture.
4. *Undistortion.* Matched pixel coordinates are corrected for lens distortion with the calibrated distortion
   model, since triangulation assumes a pinhole camera.
5. *Epipolar outlier rejection.* Because the rig is calibrated, the fundamental matrix is computed directly as
   F = K_R^-T [t]_x R K_L^-1 instead of being estimated from the matches with RANSAC. Each match's Sampson distance
   (a first-order estimate of its pixel distance from the epipolar geometry) must be below 2 px.
6. *Triangulation.* With projection matrices P_L = K_L[I | 0] and P_R = K_R[R | t], each match is triangulated by the
   linear (DLT) method [1] and converted from homogeneous coordinates, giving a 3D point in the left camera frame.
7. *Geometric filtering.* Points are kept only if they reproject within 2 px in both images (no depth-range clean-up).
8. *World frame.* The points of pair i are transformed with X_world = R_i^T (x_cam - t_i) and coloured from the left
   image.

The 49 world-frame point sets are concatenated into a single point cloud.

**Evaluation.** Each point is compared with the reference surface directly below it (nearest 1 cm grid cell). The
signed height error gives the bias (median), the RMSE, a robust spread (median absolute deviation, MAD) and the
fraction of points within 5 cm. Density is the total number of points; coverage is the fraction of 5 cm terrain
cells containing at least one point. To compare interest points and parameters, the same pipeline is re-run with
SIFT, ORB and AKAZE and with different ratio thresholds, detector thresholds and with/without CLAHE.

### 1.3 Results and Discussion *[15 marks: compare interest point types and parameters; effect on accuracy, density and types of structures covered]*

**Feature extraction and matching.** *[Figure 2 = `q1_fig02.png`.]* Figure 2 shows matches for one stereo pair.
SIFT finds 19,112 keypoints in the left image; 8,790 matches pass the ratio test and 8,688 pass the geometric checks.
Accepted matches form near-parallel lines, as expected for an 8 cm baseline. The rejected matches join unrelated
image regions.

Table 1 shows the effect of each rejection step over all 49 pairs. Without rejection, half the candidate matches
are wrong and the point cloud is unusable (RMSE 1.55 m, errors up to 70 m). The ratio test removes 54% of the
candidates and brings the median error to 1 cm, but a few thousand distinctive yet wrong matches remain, some of
them metres away (RMSE 0.23 m). The epipolar check removes only 1.1% of the remaining matches but cuts the RMSE to
0.04 m and the worst error from 61 m to 3.4 m. The reprojection check removes nothing further, because a match within
2 px of its epipolar line already reprojects within 1.4 px. The two kinds of test catch different errors. Without the ratio test, the geometric
checks alone leave an RMSE of 0.196 m, five times the final value.

*Table 1 - Effect of outlier rejection (SIFT + CLAHE, 49 pairs).*

| Stage | Points | RMSE (m) | Median abs. error (cm) | Within 5 cm | Worst error (m) |
|---|---|---|---|---|---|
| Candidate matches | 1,031,900 | 1.547 | 7.3 | 48.2% | 70.2 |
| + ratio test | 471,572 | 0.229 | 1.0 | 92.5% | 61.2 |
| + epipolar check | 466,459 | 0.040 | 1.0 | 93.4% | 3.4 |
| + reprojection check (final) | 466,459 | 0.040 | 1.0 | 93.4% | 3.4 |

**Final point cloud.** *[Figure 3 = `q1_fig03.png`, Figure 4 = `q1_fig04.png`.]* The final cloud has 466,459
points (Figure 3). It follows the reference terrain closely: the median height error is -1 mm (no systematic
bias), the MAD is 1.0 cm, 93.4% of points lie within 5 cm, and the RMSE is 4.0 cm (Figure 4). Each image covers only about
1.6 m x 1.2 m of seafloor, so the cameras imaged a narrow strip along their path, so the cloud covers 43% of the reference terrain. The reference model was built
from a larger survey and covers about 35 m², compared with about 15 m² for the cloud. Errors are not uniform:
they cluster along coral ridges and edges (Figure 4, centre). There, the 2.5D reference grid cannot represent
overhangs, and the two cameras see different sides of the structure, so occlusions and wrong matches are more
likely. The error also grows with distance from the camera (`q1_fig06.png`). This is expected: depth error grows as
Z²/(fB), about 2.9 cm per pixel of disparity at 2 m for this rig.

**Comparison of interest points and parameters.** *[Figure 5 = `q1_fig05.png`; Table 2.]*

*Table 2 - Detector and parameter comparison (same pipeline).*

| Configuration | Points | RMSE (m) | MAD (cm) | Within 5 cm | Coverage |
|---|---|---|---|---|---|
| SIFT, ratio 0.8, CLAHE (final) | 466k | 0.040 | 1.0 | 93.4% | 43% |
| SIFT, ratio 0.6 | 359k | 0.034 | 0.9 | 94.6% | 42% |
| SIFT, no ratio test | 530k | 0.196 | 1.0 | 91.8% | 44% |
| SIFT, no CLAHE | 191k | 0.039 | 0.9 | 93.3% | 38% |
| SIFT, contrastThreshold 0.01 | 572k | 0.044 | 1.0 | 92.6% | 44% |
| ORB, 1000 features | 23k | 0.048 | 1.8 | 86.3% | 16% |
| ORB, 10000 features | 223k | 0.047 | 1.8 | 87.2% | 31% |
| AKAZE, default | 329k | 0.038 | 1.0 | 92.3% | 41% |
| AKAZE, threshold 1e-4 | 665k | 0.043 | 1.1 | 91.2% | 43% |

*Interest point type.* SIFT gives the densest accurate cloud. AKAZE reaches the same accuracy (MAD 1.0 cm) with
30% fewer points and roughly a third of the computation time. ORB is the fastest but the least suitable here:
even with 10,000 features it produces half as many points as SIFT, and its MAD is almost twice as large (1.8 cm).
A likely reason is localisation: ORB keypoints are located to whole pixels on each level of an image pyramid,
while SIFT refines its keypoints to sub-pixel accuracy. Localisation error directly becomes depth error, because one
pixel of disparity is about 3 cm of depth.

*Parameters.* The parameters trade density against accuracy. A stricter ratio test (0.6) removes 23% of the points
and lowers the RMSE from 0.040 to 0.034 m. Removing the ratio test adds 14% more points but raises the RMSE five-fold.
Lowering the detector threshold (SIFT contrastThreshold 0.01, AKAZE threshold 1e-4) adds weak, low-contrast
keypoints. Density increases by 23-100%, but these keypoints are less distinctive and the RMSE rises slightly
(0.043-0.044 m). CLAHE has the largest effect on density: without it SIFT finds 2.4x fewer points (191k) at the same
accuracy, and coverage drops from 43% to 38%. The gain is largest in the dark, low-contrast parts of the images,
where water absorption has reduced the texture. Points from the darker half of the image (by local brightness)
increase 4.7x (33k to 155k), compared with 2.0x in the brighter half.

*Types of structures covered.* *[Figure 6 = `q1_fig10.png` (or put it in the appendix).]* The terrain is divided
into 5 cm cells, which are split into thirds by local slope: flat (< 20°, mostly sand and rubble), moderate
(20-37°) and steep (> 37°, coral faces and edges). Within the area the cameras imaged, SIFT
with CLAHE reconstructs almost every 5 cm cell regardless of slope (99-100%). Without CLAHE, coverage falls to 86-89%,
and the steep cells lose the most. ORB covers flat areas better than steep ones (76% vs 68% with 10,000 features;
41% vs 33% with 1,000). AKAZE covers 94-95% of every class. Accuracy depends on structure for every detector. The median error on steep cells is
about 2.5x that on flat cells (1.8 cm vs 0.7 cm for SIFT). Flat sand and rubble have fine, uniform texture that
gives many accurate matches. Steep coral faces are seen obliquely and partly occluded, and they are poorly
represented by the 2.5D reference grid.

*Limitations and improvements.* The cloud is sparse, limited to interest points. Dense matching such as semi-global
matching [3] on rectified pairs would fill the gaps. Overlapping frames observe the same surface several times but
are simply concatenated. Fusing them, or refining poses and points by bundle adjustment, would reduce noise and
correct pose errors. The accuracy is also limited by the short 8 cm baseline. Finally, part of the measured error
comes from the reference itself: a 2.5D height grid that cannot represent overhangs.

---

## Appendix A - Question 1 *[not counted in the 4 pages]*

Suggested content:
- **A.1 Additional figures:** `q1_fig01.png` (SIFT keypoints), `q1_fig07.png` (matches without the ratio test),
  `q1_fig08.png` (ORB matches), `q1_fig09.png` (AKAZE matches), `q1_fig06.png` (error vs range) and `q1_fig10.png`
  (structures) if they don't fit in the body.
- **A.2 Parameter values:** the table of settings from `README.md` (thresholds, ratio, CLAHE parameters).
- **A.3 Code overview:** the function table from `Q1_EXPLANATION.md` section 9, or a short description of
  `MainQ1.py`'s structure.
- **A.4 How to run:** `python MainQ1.py`; set `images_left_dir` / `images_right_dir`; dependencies; ~8 min runtime.

---

## Figure plan for the body

| # | File | Caption (draft) |
|---|---|---|
| 1 | *(diagram - optional)* | Stereo reconstruction pipeline applied to each image pair. |
| 2 | `q1_fig02.png` | Pair 20: random sample of accepted matches (top, green) and all matches rejected by the epipolar and reprojection checks (bottom, red). |
| 3 | `q1_fig03.png` | Reconstructed point cloud (466,459 points) in image colours with the camera path (left) and over the reference terrain (right). |
| 4 | `q1_fig04.png` | Comparison with the reference terrain: point footprint (left), signed height error (centre), error histogram (right). |
| 5 | `q1_fig05.png` | Detector and parameter comparison: matches, inlier rate, time, density, accuracy and coverage. |
| 6 | `q1_fig10.png` | Coverage of the imaged area (left) and median height error (right) on flat, moderate and steep terrain. |

---

## AI-use statement *(end of the whole report)* - draft

"Claude (Anthropic), used through Claude Code, was used to help write and debug the Python code for both questions
(`MainQ1.py`, `MainQ2.py`, `assign2_sceneclassifier.py`), to suggest analyses and figures, and to draft and
restructure parts of the report text. All code was run and checked by me, all results come from my code, and I
reviewed, edited and take responsibility for all content."
*[Adjust to match exactly how you used it.]*

---

## References *[check every reference before using it]*

[1] R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed. Cambridge University Press, 2004.

[2] D. Scharstein and R. Szeliski, "A taxonomy and evaluation of dense two-frame stereo correspondence algorithms,"
*International Journal of Computer Vision*, vol. 47, pp. 7-42, 2002.

[3] H. Hirschmüller, "Stereo processing by semiglobal matching and mutual information," *IEEE Transactions on Pattern
Analysis and Machine Intelligence*, vol. 30, no. 2, pp. 328-341, 2008.

[4] D. G. Lowe, "Distinctive image features from scale-invariant keypoints," *International Journal of Computer
Vision*, vol. 60, no. 2, pp. 91-110, 2004.

[5] E. Rublee, V. Rabaud, K. Konolige and G. Bradski, "ORB: An efficient alternative to SIFT or SURF," in *Proc.
IEEE International Conference on Computer Vision (ICCV)*, 2011.

[6] P. F. Alcantarilla, J. Nuevo and A. Bartoli, "Fast explicit diffusion for accelerated features in nonlinear
scale spaces," in *Proc. British Machine Vision Conference (BMVC)*, 2013.

[7] M. A. Fischler and R. C. Bolles, "Random sample consensus: A paradigm for model fitting with applications to image
analysis and automated cartography," *Communications of the ACM*, vol. 24, no. 6, pp. 381-395, 1981.

[8] M. Maimone, Y. Cheng and L. Matthies, "Two years of visual odometry on the Mars Exploration Rovers," *Journal of
Field Robotics*, vol. 24, no. 3, pp. 169-186, 2007.

[9] A. Geiger, P. Lenz and R. Urtasun, "Are we ready for autonomous driving? The KITTI vision benchmark suite," in
*Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2012.

[10] M. Johnson-Roberson, O. Pizarro, S. B. Williams and I. Mahon, "Generation and visualization of large-scale
three-dimensional reconstructions from underwater robotic surveys," *Journal of Field Robotics*, vol. 27, no. 1,
pp. 21-51, 2010.

[11] O. Pizarro, R. M. Eustice and H. Singh, "Large area 3-D reconstructions from underwater optical surveys," *IEEE
Journal of Oceanic Engineering*, vol. 34, no. 2, pp. 150-169, 2009.

[12] K. Zuiderveld, "Contrast limited adaptive histogram equalization," in *Graphics Gems IV*, P. Heckbert, Ed.
Academic Press, 1994, pp. 474-485.

---

## Still to do for Question 1

| Item | Status |
|---|---|
| `MainQ1.py` runs, comments, `images_left_dir` / `images_right_dir` at the top | Done |
| Point cloud plot, comparison with the reference terrain | Done |
| Detector / parameter comparison | Done |
| "Types of structures covered" analysis | Done - printed by `MainQ1.py` + `q1_fig10.png` |
| CLAHE dark/bright split | Done - printed by `MainQ1.py` |
| Before/after outlier rejection table (Table 1) | Done - printed by `MainQ1.py` |
| File name | The PDF says both `mainQN.py` (general rules) and `MainQ1.py` (Q1 section). Currently `MainQ1.py`. |
| Pickle location | `MainQ1.py` reads the three `.pkl` files from the folder above `images_left_dir`. This works with the original unzipped folder layout. If the tutor moves the image folders, the pickles must move with them. |
| Write the report | Use this draft; fit Q1 into 4 pages. |
| Appendix A (Q1) | Extra figures + settings + code overview (see above). |
| AI-use statement | At the end of the report (draft above). |
| Code zip | `SID_Assignment2.zip` with `MainQ1.py` (and Q2 files); images not needed. |
