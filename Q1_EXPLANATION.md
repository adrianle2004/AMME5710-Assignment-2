# Question 1 - How the Code Works

This document explains what `MainQ1.py` does: how it reconstructs the seafloor from the stereo images, how it
detects and rejects outlier matches, and how much each rejection step changes the result. The function reference
is at the end. See `README.md` for how to run it and the full comparison results.

---

## 1. Inputs

**Code:** [`load_pickle`](MainQ1.py#L80), [`load_stereo_pair`](MainQ1.py#L86)

| Data | Content |
|---|---|
| `images_left/` | 49 colour images, 1360x1024 |
| `images_right/` | 49 monochrome images, 1360x1024, taken at the same instants |
| `calib_stereo_diver.pkl` | `Kl, Kr` (intrinsics, f ~ 1723 px), `Dl, Dr` (lens distortion), `R, t` (right camera relative to left) |
| `camera_pose_data.pkl` | For each of the 49 frames: `R` (3x3) and `t` (3x1) of the left camera, plus the image filenames |
| `terrain_data.pkl` | Reference terrain: `height_grid` (1000x500, NaN where there is no data), `X`, `Y` |

The stereo calibration gives `t = [0, -0.080, 0.001]` m: the two cameras are **8 cm apart along the image y-axis**.
The disparity between the views is therefore mostly vertical in the image, and the epipolar lines are close to
image columns.

## 2. Coordinate frames

**Code:** [`projection_matrices`](MainQ1.py#L109), [`camera_to_world`](MainQ1.py#L249), [`TerrainModel`](MainQ1.py#L301)

- **Left camera frame:** origin at the left camera, Z along the optical axis. Every stereo pair is triangulated
  in this frame.
- **Right camera frame:** `x_right = R x_left + t` (calibration `R, t`).
- **World frame:** the pose of frame `i` maps world to camera, `x_cam = R_i X_world + t_i` (Week 5 convention).
  The code inverts it: `X_world = R_i^T (x_cam - t_i)`.

In the world frame, Z points **down**: the camera centres `-R_i^T t_i` sit at Z ~ 2.4-3.0 m, above a seafloor at
Z ~ 3.7-5.7 m, with their optical axes pointing along +Z. The reference terrain surface in this frame is
`Z = -height_grid`, laid out on `(flip(X), Y)` - the same layout the assignment's plotting code uses.

## 3. Reconstruction of one stereo pair

**Code:** [`build_point_cloud`](MainQ1.py#L273), [`reconstruct_pair`](MainQ1.py#L207), [`detect_features`](MainQ1.py#L197)

`reconstruct_pair` (with `detect_features` and `build_point_cloud`) runs these steps for each of the 49 pairs.

### 3.1 Pre-processing

**Code:** [`load_stereo_pair`](MainQ1.py#L86), [`apply_clahe`](MainQ1.py#L97), [CLAHE call in `detect_features`](MainQ1.py#L201)
The left colour image is converted to grayscale (the right image already is). CLAHE (contrast-limited adaptive
histogram equalisation, clip limit 2.0, 8x8 tiles) is applied to both. Underwater images are low contrast, and a
colour camera and a mono camera respond differently to the same scene; local equalisation makes the two views look
more alike and brings out texture in dark regions. Without CLAHE the final cloud has 191k points instead of 466k.

### 3.2 Feature detection

**Code:** [`create_detector`](MainQ1.py#L159), [`detect_features`](MainQ1.py#L197)
SIFT (`cv2.SIFT_create`, default parameters) finds keypoints and computes a 128-value descriptor for each. With
CLAHE this gives ~21,000 keypoints per image (1.03 million left keypoints over the 49 pairs). The comparison
experiment swaps SIFT for ORB or AKAZE.

### 3.3 Matching

**Code:** [`match_ratio_test`](MainQ1.py#L171)
Every left descriptor is compared with every right descriptor (`cv2.BFMatcher`, L2 distance for SIFT). For each
left keypoint, the two closest right descriptors are kept (`knnMatch`, k=2). The closest one is the **candidate
match**. At this point every left keypoint has a candidate, whether or not the point is actually visible in the
right image - so many candidates are wrong.

### 3.4 Undistortion

**Code:** [undistortion in `reconstruct_pair`](MainQ1.py#L230)
The matched pixel coordinates are corrected for lens distortion with
`cv2.undistortPoints(pts, K, D, None, K)`. The distortion is large (k1 ~ 0.15, k2 ~ 0.6 - partly from the
underwater housing), and triangulation and the epipolar check assume an ideal pinhole camera. Passing `K` as the
last argument keeps the result in pixel units.

### 3.5 Triangulation

**Code:** [`projection_matrices`](MainQ1.py#L109), [`triangulate`](MainQ1.py#L184)
The projection matrices are

```
Pleft  = Kl [I | 0]        (left camera at the origin)
Pright = Kr [R | t]        (right camera, from the calibration)
```

`cv2.triangulatePoints(Pleft, Pright, pts_left, pts_right)` solves, for each match, the linear system (DLT) for the
3D point whose projections best agree with the two pixel positions. The result is a homogeneous 4-vector; dividing
by its 4th coordinate gives `(X, Y, Z)` in the left camera frame. Z is the distance from the camera, 1.2-3 m here.

### 3.6 Transform to the world frame

**Code:** [`camera_to_world`](MainQ1.py#L249), [call in `build_point_cloud`](MainQ1.py#L284)
Each pair's surviving points are moved into the world frame with that frame's pose: `X_world = R_i^T (x_cam - t_i)`.
Each point is also given the RGB colour of the left-image pixel it came from.

### 3.7 Merging

**Code:** [stacking in `build_point_cloud`](MainQ1.py#L295)
The world-frame points of all 49 pairs are stacked into one point cloud (466,459 points). Neighbouring frames
overlap heavily, so most of the seafloor is seen in several pairs; no merging or averaging of duplicates is done.

---

## 4. Outlier detection and rejection

**Code:** [`reconstruct_pair`](MainQ1.py#L207)

A wrong match triangulates to a wrong 3D point, and a single wrong match can land metres away from the seafloor.
The code rejects outliers in three stages. A match must pass **all** of them to become a point in the cloud.

### Stage 1 - Lowe's ratio test (appearance)

**Code:** [`match_ratio_test`](MainQ1.py#L171)
For each left keypoint, compare the distance to its best right descriptor `d1` with the distance to the second-best
`d2`. Keep the match only if

```
d1 < 0.8 * d2
```

If the best candidate is not clearly better than the runner-up, the descriptor is ambiguous: this happens on
repetitive texture (coral branches, rubble, sand) and for points that are not visible in the right image at all.
This test uses appearance only, no geometry.

### Stage 2 - Epipolar constraint (geometry)

**Code:** [`skew`](MainQ1.py#L120), [`calibrated_fundamental_matrix`](MainQ1.py#L128), [`sampson_distance`](MainQ1.py#L136), [check in `reconstruct_pair`](MainQ1.py#L234)
For a correct match, the right point must lie on the **epipolar line** of the left point. Because the stereo rig is
calibrated, the fundamental matrix is computed directly from the calibration instead of being estimated from the
matches:

```
E = [t]x R                  (essential matrix; [t]x is the cross-product matrix of t)
F = Kr^-T E Kl^-1           (fundamental matrix, for undistorted pixels)
```

For each undistorted match `(xl, xr)` the code computes the **Sampson distance** - a first-order estimate of how
far, in pixels, the pair is from satisfying `xr^T F xl = 0`, using the epipolar lines in both images:

```
d = (xr^T F xl)^2 / ( (F xl)_1^2 + (F xl)_2^2 + (F^T xr)_1^2 + (F^T xr)_2^2 )
```

The match is kept if `sqrt(d) < 2 px`. Using the calibration's F (instead of RANSAC, as in the tutorial) means the
geometry is exact and fixed, the result is deterministic, and it does not fail on a nearly flat seafloor (a
degenerate case for estimating F from points).

### Stage 3 - Reprojection error

**Code:** [`reprojection_error`](MainQ1.py#L148), [check in `reconstruct_pair`](MainQ1.py#L241)
After triangulation, each 3D point is projected back into both cameras with `Pleft` and `Pright`. The match is kept
if the reprojection error is below 2 px in **both** images. This catches matches that the linear triangulation
cannot fit consistently.

There is deliberately **no depth-range filter** after triangulation: the aim is to evaluate how well the
matched points reconstruct the terrain, not to clean up the final cloud. A few matches that lie on the epipolar
line but at the wrong position along it therefore remain (worst error 3.4 m).

---

## 5. Before vs. after outlier rejection

**Code:** [`rejection_stages`](MainQ1.py#L573), [`print_rejection_stages`](MainQ1.py#L613), [`plot_example_matches`](MainQ1.py#L385)

The numbers below are printed by `MainQ1.py` (over all 49 pairs, final settings: SIFT + CLAHE). For every pair,
`rejection_stages` triangulates **all** candidate matches, applies the stages one at a time, and compares each
cumulative stage with the reference terrain, so the table shows what each stage removes.

| Stage | Points kept | Removed at this stage | RMSE vs terrain | Median abs. error | Within 5 cm | Worst error |
|---|---|---|---|---|---|---|
| Candidate matches (no rejection) | 1,031,900 | - | 1.547 m | 7.3 cm | 48.2% | 70.2 m |
| After ratio test | 471,572 | 560,328 (54%) | 0.229 m | 1.0 cm | 92.5% | 61.2 m |
| After epipolar check | 466,459 | 5,113 (1.1%) | 0.040 m | 1.0 cm | 93.4% | 3.4 m |
| **After reprojection check (final)** | **466,459** | **0** | **0.040 m** | **1.0 cm** | **93.4%** | **3.4 m** |

What this shows:
- **Without rejection the cloud is unusable:** half of the candidate matches are wrong, the RMSE is 1.5 m, and
  points reach 70 m from the seafloor.
- **The ratio test removes most of the outliers** (54% of candidates). The median error drops from 7.3 cm to 1 cm.
  But the RMSE is still 0.23 m, because a few thousand wrong-but-distinctive matches survive, some metres away.
- **The epipolar check removes those remaining gross outliers.** It only removes 1.1% of the matches, but it cuts
  the RMSE from 0.23 m to 0.04 m and the worst error from 61 m to 3.4 m.
- **The reprojection check removes nothing here.** A match that passes the 2 px Sampson test already reprojects
  within 1.4 px (median 0.26 px), so this stage is only a safety net.

The two geometric stages also work without the ratio test, but less well: epipolar + reprojection alone keep
530k points with RMSE 0.196 m, five times the final value (the "SIFT no ratio test" configuration). The ratio test and
the epipolar check catch different errors - ambiguous appearance vs. inconsistent geometry - so both are needed.

Before the ratio test only 51% of candidates are within 2 px of their epipolar line; after it, 98.9% are.
The ratio test alone therefore gets the matches mostly right, and the epipolar check cleans up the rest.

**Example - pair 20** (`q1_fig02.png`): 19,112 left and 18,627 right keypoints -> 19,112 candidates ->
8,790 after the ratio test -> 8,688 after the epipolar check -> 8,688 after the reprojection check.

How to read `q1_fig02.png`: each panel shows the left (colour) image and the right (mono) image side by side, and
each line joins a left keypoint to the right keypoint it was matched to.
- **Top (green):** 150 randomly chosen matches out of the 8,688 accepted ones. Because the two cameras only differ
  by an 8 cm shift, a correct match lands at almost the same place in the other image, so all green lines are
  nearly parallel and about one image-width long. Small differences in their slope are the disparity, which is
  what the triangulation turns into depth.
- **Bottom (red):** all 102 matches that passed the ratio test but were then rejected by the geometric checks.
  They join points in unrelated places, so the lines cross at random angles - these would have become wrong 3D
  points. The matches already removed by the ratio test are not drawn here.
 `q1_fig07.png` shows the same for the run without the ratio test, where far more
matches are rejected by geometry alone.

---

## 6. Comparison with the reference terrain

**Code:** [`TerrainModel`](MainQ1.py#L301), [`TerrainModel.lookup`](MainQ1.py#L312), [`compare_to_terrain`](MainQ1.py#L325), [`plot_terrain_comparison`](MainQ1.py#L462), [`plot_depth_error_by_range`](MainQ1.py#L541)

`compare_to_terrain` looks up the reference surface Z directly below each point (nearest 1 cm grid cell) and
computes the signed height error `point Z - terrain Z`. It reports:
- **RMSE** - overall error, sensitive to outliers.
- **Median** - the bias; -1 mm for the final cloud, so there is no systematic offset between the reconstruction
  and the reference.
- **MAD** (median absolute deviation) - a robust spread that ignores a few outliers; 1 cm for the final cloud.
- **% within 5 cm** of the terrain.
- **Coverage** - the fraction of 5 cm x 5 cm terrain cells (with reference data) that contain at least one point;
  43% for the final cloud. This measures how much of the surface is reconstructed, independently of point count.

The error is largest on coral edges and overhangs (`q1_fig04.png`). There, the 2.5D reference grid (one height per
cell) cannot represent the shape, and the two cameras see different sides of the structure. The error also grows
with distance from the camera (`q1_fig06.png`): with an 8 cm baseline, one pixel of disparity corresponds to about
3 cm of depth at 2 m (`Z^2 / (f B)`).

## 7. Detector and parameter comparison

**Code:** [`CONFIG`](MainQ1.py#L45), [`COMPARISON_CONFIGS`](MainQ1.py#L56), [`variant`](MainQ1.py#L48), [`FeatureCache`](MainQ1.py#L254), [`run_config`](MainQ1.py#L736), [`plot_comparison`](MainQ1.py#L498)

After the main reconstruction, the whole pipeline is re-run for each entry of `COMPARISON_CONFIGS`: a different ratio
threshold, no ratio test, no CLAHE, a lower SIFT contrast threshold, ORB with 1000 or 10000 features, and AKAZE
with two thresholds. Each run builds its own world point cloud and is evaluated the same way. Features are detected
once per detector setting and reused (`FeatureCache`) when only the matching changes. Results are in `README.md`
and `q1_fig05.png`.

## 8. Further analysis: structures covered and the effect of CLAHE

These are printed after the comparison results.

### 8.1 Types of structures covered

**Code:** [`terrain_slope_classes`](MainQ1.py#L623), [`structure_coverage`](MainQ1.py#L642), [`print_structure_coverage`](MainQ1.py#L670), [`plot_structure_coverage`](MainQ1.py#L680)

The reference terrain is divided into 5 cm cells. Each cell gets a slope from the gradient of the cell-mean heights,
and the cells are split into thirds:
- **flat** (< 20 deg): mostly sand and rubble,
- **moderate** (20-37 deg),
- **steep** (> 37 deg): coral faces and edges.

For SIFT, SIFT without CLAHE, ORB 1000, ORB 10000 and AKAZE, the code reports two things per slope class:
- **Coverage:** the % of cells containing a point. It is measured only inside the area the cameras imaged (cells
  reconstructed by at least one of these configurations), so it shows what each detector misses, not what the
  cameras never saw.
- **Median |height error|** of the points in that class.

| Configuration | Coverage flat / moderate / steep | Median error flat / moderate / steep |
|---|---|---|
| SIFT (CONFIG) | 99.6% / 99.7% / 99.1% | 0.7 / 0.9 / 1.8 cm |
| SIFT no CLAHE | 89.4% / 88.8% / 86.0% | 0.7 / 0.9 / 1.8 cm |
| ORB 1000 | 40.6% / 34.7% / 32.5% | 1.7 / 1.7 / 2.3 cm |
| ORB 10000 | 76.2% / 71.4% / 67.6% | 1.7 / 1.7 / 2.3 cm |
| AKAZE default | 94.9% / 95.3% / 94.3% | 0.8 / 1.0 / 1.9 cm |

- **Steep coral is harder for every detector:** its median error is about 2.5x that of flat ground for SIFT.
  Steep faces are seen obliquely, are partly occluded in one view, and are poorly represented by the 2.5D
  reference grid.
- **ORB misses steep structures most** (76% flat vs 68% steep coverage with 10,000 features). SIFT with CLAHE
  covers almost every cell regardless of slope.
- **Without CLAHE** SIFT loses about 10-13% of the imaged cells, most of them on steep terrain.

The bar chart is `q1_fig10.png`.

### 8.2 Effect of CLAHE on dark and bright regions

**Code:** [`brightness_split`](MainQ1.py#L700), [`print_brightness_split`](MainQ1.py#L715)

Each point is assigned the local brightness (31x31 mean of the raw left grayscale image) at the pixel it came from.
Points are split at the median local brightness over all images.

| | Points from dark regions | Points from bright regions |
|---|---|---|
| SIFT with CLAHE | 155,262 | 311,180 |
| SIFT without CLAHE | 33,367 | 157,831 |
| Gain from CLAHE | x4.7 | x2.0 |

CLAHE helps most where the water has darkened the image and reduced the texture.

## 9. Figures

**Code:** [`plot_keypoints`](MainQ1.py#L369), [`plot_example_matches`](MainQ1.py#L385), [`plot_point_cloud`](MainQ1.py#L413), [`plot_terrain_comparison`](MainQ1.py#L462), [`plot_comparison`](MainQ1.py#L498), [`plot_depth_error_by_range`](MainQ1.py#L541), [`plot_structure_coverage`](MainQ1.py#L680), [`show_figures`](MainQ1.py#L724), [`main`](MainQ1.py#L746)

| Figure | Content |
|---|---|
| `q1_fig01.png` | SIFT keypoints in both images of pair 20 |
| `q1_fig02.png` | Pair 20: sample of accepted (green) and rejected (red) matches |
| `q1_fig03.png` | Left: our reconstruction in true colour with the camera path. Right: the same points coloured by depth, over the reference terrain (grey surface). The points lie on the surface (~1 cm error), so the semi-transparent surface hides about half of them; `q1_fig04.png` shows the agreement more clearly |
| `q1_fig04.png` | Point footprint on the terrain, map of height errors, error histogram |
| `q1_fig05.png` | Bar charts of the detector/parameter comparison |
| `q1_fig06.png` | Median error vs. distance from the camera |
| `q1_fig07-09.png` | Accepted/rejected matches for SIFT without ratio test, ORB 10000, AKAZE |
| `q1_fig10.png` | Coverage and median error by terrain slope (flat / moderate / steep) for SIFT, SIFT without CLAHE, ORB and AKAZE |

All figures are shown on screen while the script runs and saved to `figures_q1/` as soon as they are drawn, so
closing a window early does not lose the figure. The figure windows only respond (redraw, zoom) when the script
pauses to draw - after the main figures and at the end - so they appear frozen while the comparison is computing.

---

## 10. Function reference

### Settings
- `images_left_dir`, `images_right_dir` - image folders (changed by the tutor); the pickle files are read from their parent folder.
- `CONFIG` - the pipeline settings (detector, detector parameters, CLAHE on/off, ratio threshold).
- `COMPARISON_CONFIGS` - variations of `CONFIG` compared in the experiment.
- Thresholds (`EPIPOLAR_THRESH`, `REPROJ_THRESH`) and plot options.

### Functions

| Function | What it does |
|---|---|
| [`variant(**changes)`](MainQ1.py#L48) | Copies `CONFIG` with some settings changed (used for the comparison). |
| [`load_pickle(path)`](MainQ1.py#L80) | Loads a pickle file (calibration, poses, terrain). |
| [`load_stereo_pair(idx, poses)`](MainQ1.py#L86) | Loads pair `idx`; returns the left image in RGB and grayscale, and the right image in grayscale. |
| [`apply_clahe(gray)`](MainQ1.py#L97) | Applies CLAHE contrast equalisation. |
| [`projection_matrices(calib)`](MainQ1.py#L109) | Builds `Pleft = Kl[I|0]` and `Pright = Kr[R|t]`. |
| [`skew(v)`](MainQ1.py#L120) | Returns the 3x3 cross-product matrix `[v]x`. |
| [`calibrated_fundamental_matrix(calib)`](MainQ1.py#L128) | Computes F from the calibration: `F = Kr^-T [t]x R Kl^-1`. |
| [`sampson_distance(F, pts_l, pts_r)`](MainQ1.py#L136) | Pixel distance of each match from the epipolar geometry defined by F. |
| [`reprojection_error(P, X, pts)`](MainQ1.py#L148) | Pixel distance between projected 3D points and the observed points. |
| [`create_detector(name, params)`](MainQ1.py#L159) | Creates an ORB, SIFT or AKAZE detector and returns it with its matching norm. |
| [`match_ratio_test(des_l, des_r, norm, ratio)`](MainQ1.py#L171) | k-NN matching with Lowe's ratio test. |
| [`triangulate(Pleft, Pright, pts_left, pts_right)`](MainQ1.py#L184) | Triangulates matches and returns Nx3 points in the left camera frame. |
| [`detect_features(gray_l, gray_r, cfg)`](MainQ1.py#L197) | Detects keypoints and descriptors in both images (with CLAHE if configured). |
| [`reconstruct_pair(features, cfg, calib, geom)`](MainQ1.py#L207) | For one pair: match, undistort, epipolar check, triangulate, reprojection filter. Returns the 3D points and statistics. |
| [`camera_to_world(points_cam, R, t)`](MainQ1.py#L249) | Converts camera-frame points to the world frame: `X_world = R^T (x_cam - t)`. |
| [`FeatureCache`](MainQ1.py#L254) | Reuses detected features between configurations that use the same detector settings. |
| [`build_point_cloud(cfg, calib, poses, images, cache)`](MainQ1.py#L273) | Runs all 49 pairs, transforms them to the world frame and merges them into one coloured point cloud. |
| [`TerrainModel`](MainQ1.py#L301) | Holds the reference terrain; `lookup(x, y)` returns the terrain Z below a world point. |
| [`compare_to_terrain(points, terrain_model)`](MainQ1.py#L325) | Height error of every point vs. the terrain, with RMSE, MAD, % within 5 cm and coverage. |
| [`print_summary(label, s)`](MainQ1.py#L352) | Prints one line of the statistics. |
| [`subsample(n, max_n)`](MainQ1.py#L362) | Random subset of indices for plotting. |
| [`plot_keypoints(...)`](MainQ1.py#L369) | Plots the keypoints in the left and right images of one pair. |
| [`plot_example_matches(...)`](MainQ1.py#L385) | Plots a sample of inlier (green) and rejected (red) matches for one pair. |
| [`plot_point_cloud(...)`](MainQ1.py#L413) | 3D plot of the point cloud with the camera path, and the cloud over the reference terrain. |
| [`plot_terrain_comparison(...)`](MainQ1.py#L462) | Point footprint on the terrain, spatial height-error map and error histogram. |
| [`plot_comparison(results, title)`](MainQ1.py#L498) | Bar charts comparing a set of configurations. |
| [`plot_depth_error_by_range(results)`](MainQ1.py#L541) | Median height error vs. distance from the camera. |
| [`rejection_stages(...)`](MainQ1.py#L573) | Triangulates all candidate matches of every pair and evaluates the cloud after each rejection stage (ratio, epipolar, reprojection). |
| [`print_rejection_stages(rows)`](MainQ1.py#L613) | Prints the stage-by-stage table. |
| [`terrain_slope_classes(terrain_model)`](MainQ1.py#L623) | Classifies each 5 cm terrain cell as flat, moderate or steep by its slope (thirds). |
| [`structure_coverage(results, terrain_model)`](MainQ1.py#L642) | Coverage of the imaged area and median error per slope class, for each configuration. |
| [`print_structure_coverage(...)`](MainQ1.py#L670) | Prints the structure table. |
| [`plot_structure_coverage(table, edges)`](MainQ1.py#L680) | Bar charts of coverage and error per slope class (`q1_fig10.png`). |
| [`brightness_split(results, images)`](MainQ1.py#L700) | Counts points coming from dark vs bright image regions. |
| [`print_brightness_split(rows)`](MainQ1.py#L715) | Prints the dark/bright counts and the CLAHE gain. |
| [`show_figures(saved)`](MainQ1.py#L724) | Saves every new figure to `figures_q1/` immediately, then draws all figures so they appear while the script is still running. |
| [`run_config(...)`](MainQ1.py#L736) | Builds and evaluates the point cloud for one configuration. |
| [`main()`](MainQ1.py#L746) | Runs everything: the main reconstruction, the stage-by-stage rejection table, the detector/parameter comparison, the structure and brightness analyses; prints the results, displays every figure and saves them to `figures_q1/`. |

