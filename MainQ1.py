"""
AMME5710 Assignment 2 - Question 1
3D Reconstruction using Underwater Stereo Vision

Follows the structure of the Week 5 tutorial stereo pipeline. For each of the
49 stereo pairs: detect features, match them, reject outliers with the
epipolar constraint, undistort, triangulate into 3D points in the left camera
frame, and transform them into the world frame using the camera pose. All
pairs are merged into one point cloud, which is plotted and compared with the
reference terrain. The same pipeline is then re-run with different detectors
and parameters for comparison.

Dependencies: numpy, opencv-python (>= 4.4 for SIFT), matplotlib
"""

import os
import pickle
import time
import warnings

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)

# ---------------------------------------------------------------------------
# Paths (change these to the location of the data on your computer)
# ---------------------------------------------------------------------------
images_left_dir = 'assignment2_stereodata/assignment2_stereodata/images_left'
images_right_dir = 'assignment2_stereodata/assignment2_stereodata/images_right'

# The pickle files are assumed to sit in the folder above the image folders
data_dir = os.path.dirname(os.path.normpath(images_left_dir))
calib_file = os.path.join(data_dir, 'calib_stereo_diver.pkl')
pose_file = os.path.join(data_dir, 'camera_pose_data.pkl')
terrain_file = os.path.join(data_dir, 'terrain_data.pkl')

# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------
#   detector   'ORB' | 'SIFT' | 'AKAZE'
#   params     keyword arguments for the OpenCV detector constructor
#   clahe      True -> apply CLAHE before feature detection
#   ratio      Lowe ratio test threshold (1.0 = no ratio test)
CONFIG = {'detector': 'SIFT', 'params': {}, 'clahe': True, 'ratio': 0.8}


def variant(**changes):
    """Copy of CONFIG with some settings changed."""
    cfg = dict(CONFIG)
    cfg.update(changes)
    return cfg


# Detector / parameter variations compared against CONFIG
COMPARISON_CONFIGS = [
    ('SIFT ratio=0.6',       variant(ratio=0.6)),
    ('SIFT no ratio test',   variant(ratio=1.0)),
    ('SIFT no CLAHE',        variant(clahe=False)),
    ('SIFT contrast=0.01',   variant(params={'contrastThreshold': 0.01})),
    ('ORB 1000',             variant(detector='ORB', params={'nfeatures': 1000})),
    ('ORB 10000',            variant(detector='ORB', params={'nfeatures': 10000})),
    ('AKAZE default',        variant(detector='AKAZE', params={})),
    ('AKAZE thresh=1e-4',    variant(detector='AKAZE', params={'threshold': 1e-4})),
]

# ---------------------------------------------------------------------------
# Other settings
# ---------------------------------------------------------------------------
EPIPOLAR_THRESH = 2.0       # max Sampson distance to calibrated epipolar line [px]
REPROJ_THRESH = 2.0         # max triangulation reprojection error [px]
EXAMPLE_PAIR = 20           # index of the stereo pair used for example plots
N_DRAW_MATCHES = 150        # number of (random) matches drawn in match plots
FIGURE_DIR = 'figures_q1'   # figures are also saved here for the report


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_pickle(path):
    """Load a python pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_stereo_pair(idx, poses):
    """Return (left colour image [RGB], left gray, right gray) for pair idx.
    The left camera is colour and the right camera is monochrome, so the
    left image is converted to grayscale for feature detection."""
    left = cv2.imread(os.path.join(images_left_dir, poses['filenames_left'][idx]), cv2.IMREAD_COLOR)
    right = cv2.imread(os.path.join(images_right_dir, poses['filenames_right'][idx]), cv2.IMREAD_GRAYSCALE)
    left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    return left_rgb, left_gray, right


def apply_clahe(gray):
    """Contrast-limited adaptive histogram equalisation (CLAHE).
    Underwater images are low contrast and the colour/mono sensors respond
    differently; CLAHE makes the local appearance of the two views more alike
    and produces more features in dark regions."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ---------------------------------------------------------------------------
# Stereo geometry
# ---------------------------------------------------------------------------
def projection_matrices(calib):
    """Projection matrices for triangulating points in the left camera's
    frame of reference:
        Pleft  = Kleft  [I | 0]
        Pright = Kright [R | t]
    where (R, t) is the pose of the right camera relative to the left."""
    Pleft = calib['Kl'] @ np.hstack((np.eye(3), np.zeros((3, 1))))
    Pright = calib['Kr'] @ np.hstack((calib['R'], calib['t'].reshape(3, 1)))
    return Pleft, Pright


def skew(v):
    """3x3 skew-symmetric (cross product) matrix [v]x of a 3-vector."""
    v = np.asarray(v).ravel()
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def calibrated_fundamental_matrix(calib):
    """Fundamental matrix from the known stereo calibration:
        E = [t]x R,   F = Kr^-T E Kl^-1
    valid for undistorted pixel coordinates."""
    E = skew(calib['t']) @ calib['R']
    return np.linalg.inv(calib['Kr']).T @ E @ np.linalg.inv(calib['Kl'])


def sampson_distance(F, pts_l, pts_r):
    """First-order geometric distance (pixels) of each correspondence to the
    epipolar geometry defined by F. pts are Nx2 undistorted pixels."""
    xl = np.hstack([pts_l, np.ones((len(pts_l), 1))])
    xr = np.hstack([pts_r, np.ones((len(pts_r), 1))])
    Fxl = xl @ F.T          # epipolar lines in the right image
    Ftxr = xr @ F           # epipolar lines in the left image
    num = np.sum(xr * Fxl, axis=1) ** 2
    den = Fxl[:, 0] ** 2 + Fxl[:, 1] ** 2 + Ftxr[:, 0] ** 2 + Ftxr[:, 1] ** 2
    return np.sqrt(num / den)


def reprojection_error(P, X, pts):
    """Pixel distance between projected 3D points X (Nx3) and pts (Nx2)."""
    Xh = np.hstack([X, np.ones((len(X), 1))])
    proj = Xh @ P.T
    proj = proj[:, :2] / proj[:, 2:3]
    return np.linalg.norm(proj - pts, axis=1)


# ---------------------------------------------------------------------------
# Feature extraction and matching
# ---------------------------------------------------------------------------
def create_detector(name, params):
    """Create an OpenCV feature detector/descriptor and its matching norm
    (Hamming for the binary ORB/AKAZE descriptors, L2 for SIFT)."""
    if name == 'ORB':
        return cv2.ORB_create(**params), cv2.NORM_HAMMING
    if name == 'SIFT':
        return cv2.SIFT_create(**params), cv2.NORM_L2
    if name == 'AKAZE':
        return cv2.AKAZE_create(**params), cv2.NORM_HAMMING
    raise ValueError('Unknown detector: ' + name)


def match_ratio_test(des_l, des_r, norm, ratio):
    """k-NN (k=2) brute-force matching followed by Lowe's ratio
    test: keep a match only if it is clearly better than the second-best
    candidate. This rejects ambiguous matches on repetitive texture (coral
    branches, sand)."""
    bf = cv2.BFMatcher(norm)
    good = []
    for pair in bf.knnMatch(des_l, des_r, k=2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])
    return good


def triangulate(Pleft, Pright, pts_left, pts_right):
    """Triangulate matched (undistorted) pixel coordinates
    and convert from homogeneous 4-vectors to 3D by dividing by the 4th
    coordinate. Returns Nx3 points in the left camera frame."""
    points_3D = cv2.triangulatePoints(Pleft, Pright,
                                      pts_left.T.astype(np.float64), pts_right.T.astype(np.float64))
    points_3D = points_3D / points_3D[3]
    return points_3D[:3, :].T


# ---------------------------------------------------------------------------
# Per-pair reconstruction
# ---------------------------------------------------------------------------
def detect_features(gray_l, gray_r, cfg):
    """Detect keypoints and descriptors in both images of a pair."""
    detector, norm = create_detector(cfg['detector'], cfg['params'])
    if cfg['clahe']:
        gray_l, gray_r = apply_clahe(gray_l), apply_clahe(gray_r)
    kp_l, des_l = detector.detectAndCompute(gray_l, None)  # 2nd arg: optional mask
    kp_r, des_r = detector.detectAndCompute(gray_r, None)
    return kp_l, des_l, kp_r, des_r, norm


def reconstruct_pair(features, cfg, calib, geom):
    """Match, filter and triangulate one stereo pair.
    features = output of detect_features. Returns a dict with the 3D points in
    the left camera frame, their left-image pixel positions and statistics."""
    kp_l, des_l, kp_r, des_r, norm = features
    Pleft, Pright, F_cal = geom

    # --- matching ---
    if des_l is None or des_r is None or len(des_l) < 2 or len(des_r) < 2:
        matches = []
    else:
        matches = match_ratio_test(des_l, des_r, norm, cfg['ratio'])

    out = {'n_kp_left': len(kp_l), 'n_kp_right': len(kp_r), 'n_matches': len(matches),
           'kp_l': kp_l, 'kp_r': kp_r, 'matches': matches, 'inlier_matches': [],
           'points_cam': np.zeros((0, 3)), 'pix_left': np.zeros((0, 2))}
    if len(matches) == 0:
        return out

    pts1 = np.float32([kp_l[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp_r[m.trainIdx].pt for m in matches])

    # --- undistort (triangulation assumes an ideal pinhole camera) ---
    und1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), calib['Kl'], calib['Dl'], None, calib['Kl']).reshape(-1, 2)
    und2 = cv2.undistortPoints(pts2.reshape(-1, 1, 2), calib['Kr'], calib['Dr'], None, calib['Kr']).reshape(-1, 2)

    # --- epipolar outlier rejection (F from the calibration) ---
    keep = sampson_distance(F_cal, und1, und2) < EPIPOLAR_THRESH

    # --- triangulation in the left camera frame ---
    X = triangulate(Pleft, Pright, und1, und2)

    # --- reject points with large reprojection error ---
    err = np.maximum(reprojection_error(Pleft, X, und1), reprojection_error(Pright, X, und2))
    keep &= err < REPROJ_THRESH

    out['inlier_matches'] = [m for m, k in zip(matches, keep) if k]
    out['points_cam'] = X[keep]
    out['pix_left'] = pts1[keep]
    return out


def camera_to_world(points_cam, R, t):
    """Week 5 convention: x_cam = R X_world + t  =>  X_world = R^T (x_cam - t)."""
    return (R.T @ (points_cam.T - t.reshape(3, 1))).T


class FeatureCache:
    """Keeps the detected features of the most recent detector setting, so
    configurations that only change matching/filtering do not re-detect.
    Only one setting is cached at a time to limit memory use."""

    def __init__(self):
        self.key, self.data, self.time = None, None, None

    def get(self, images, cfg):
        key = (cfg['detector'], repr(sorted(cfg['params'].items())), cfg['clahe'])
        if key != self.key:
            self.key, self.data, self.time = key, [], []
            for _, gray_l, gray_r in images:
                t0 = time.time()
                self.data.append(detect_features(gray_l, gray_r, cfg))
                self.time.append(time.time() - t0)
        return self.data, self.time


def build_point_cloud(cfg, calib, poses, images, cache, verbose=False):
    """Run the stereo pipeline on every pair and merge the results into one
    world-frame point cloud. Returns (points Nx3, colours Nx3, per-pair stats)."""
    Pleft, Pright = projection_matrices(calib)
    geom = (Pleft, Pright, calibrated_fundamental_matrix(calib))
    features, det_times = cache.get(images, cfg)
    all_pts, all_cols, stats = [], [], []
    for i in range(len(images)):
        t0 = time.time()
        res = reconstruct_pair(features[i], cfg, calib, geom)
        res['time'] = det_times[i] + time.time() - t0   # detection + matching + triangulation
        pw = camera_to_world(res['points_cam'], poses['R'][:, :, i], poses['t'][:, i])
        # colour each 3D point with the left image pixel it was observed at
        px = np.round(res['pix_left']).astype(int)
        cols = images[i][0][px[:, 1], px[:, 0]] / 255.0 if len(px) else np.zeros((0, 3))
        all_pts.append(pw)
        all_cols.append(cols)
        stats.append(res)
        if verbose:
            print('  pair %2d: kp L/R %5d/%5d  matches %5d  inliers %5d  (%.2fs)'
                  % (i, res['n_kp_left'], res['n_kp_right'], res['n_matches'],
                     len(res['points_cam']), res['time']))
    return np.vstack(all_pts), np.vstack(all_cols), stats


# ---------------------------------------------------------------------------
# Comparison with the reference terrain
# ---------------------------------------------------------------------------
class TerrainModel:
    """Reference terrain on a regular grid. Following the plotting code given
    in the assignment, grid cell [i, j] is located at (X=flip(X)[j], Y=Y[i])
    and the surface in the world frame (Z pointing down, same as the camera
    poses) is at Z = -height_grid[i, j]."""

    def __init__(self, terrain):
        self.x = np.flip(np.asarray(terrain['X']).ravel())
        self.y = np.asarray(terrain['Y']).ravel()
        self.z = -np.asarray(terrain['height_grid'], dtype=float)

    def lookup(self, px, py):
        """Nearest grid value of the reference surface at world (px, py).
        Returns NaN outside the grid or where the reference has no data."""
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        j = np.round((px - self.x[0]) / dx).astype(int)
        i = np.round((py - self.y[0]) / dy).astype(int)
        inside = (i >= 0) & (i < len(self.y)) & (j >= 0) & (j < len(self.x))
        out = np.full(len(px), np.nan)
        out[inside] = self.z[i[inside], j[inside]]
        return out, i, j, inside


def compare_to_terrain(points, terrain_model):
    """Signed height error (point Z - reference Z) and summary statistics.
    Coverage = number of 5 cm terrain cells containing at least one point."""
    ref, i, j, _ = terrain_model.lookup(points[:, 0], points[:, 1])
    err = points[:, 2] - ref
    valid = ~np.isnan(err)
    e = err[valid]
    cell = 5    # 5 grid cells of 1 cm = 5 cm
    occupied = set(zip(i[valid] // cell, j[valid] // cell))
    zc = terrain_model.z[:(terrain_model.z.shape[0] // cell) * cell, :(terrain_model.z.shape[1] // cell) * cell]
    has_data = ~np.isnan(zc.reshape(zc.shape[0] // cell, cell, zc.shape[1] // cell, cell)).all(axis=(1, 3))
    summary = {
        'n_points': len(points),
        'n_compared': int(valid.sum()),
        'mean': float(np.mean(e)) if len(e) else np.nan,
        'median': float(np.median(e)) if len(e) else np.nan,
        'std': float(np.std(e)) if len(e) else np.nan,
        'rmse': float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan,
        'mad': float(np.median(np.abs(e - np.median(e)))) if len(e) else np.nan,
        'median_abs': float(np.median(np.abs(e))) if len(e) else np.nan,
        'within_5cm': float(np.mean(np.abs(e) < 0.05)) if len(e) else np.nan,
        'occupied_cells_5cm': len(occupied),
        'reference_cells_5cm': int(has_data.sum()),
    }
    return err, summary


def print_summary(label, s):
    print('%-26s points %6d | compared %6d | median %+.3f m | RMSE %7.3f m | '
          'MAD %.3f m | |err|<5cm %5.1f%% | 5cm cells %d'
          % (label, s['n_points'], s['n_compared'], s['median'], s['rmse'],
             s['mad'], 100 * s['within_5cm'], s['occupied_cells_5cm']))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def subsample(n, max_n, seed=0):
    """Reproducible random subset of indices, so plots stay readable/fast."""
    if n <= max_n:
        return np.arange(n)
    return np.random.RandomState(seed).choice(n, max_n, replace=False)


def plot_keypoints(images, stats, idx, title):
    """Keypoints detected in the left and right images of one pair."""
    left_rgb, _, gray_r = images[idx]
    s = stats[idx]
    kl = cv2.drawKeypoints(left_rgb, s['kp_l'], None, color=(0, 255, 0))
    kr = cv2.drawKeypoints(cv2.cvtColor(gray_r, cv2.COLOR_GRAY2RGB), s['kp_r'], None, color=(0, 255, 0))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].imshow(kl)
    ax[0].set_title('%s left: %d keypoints' % (title, len(s['kp_l'])))
    ax[1].imshow(kr)
    ax[1].set_title('%s right: %d keypoints' % (title, len(s['kp_r'])))
    for a in ax:
        a.set_axis_off()
    fig.tight_layout()


def plot_example_matches(images, stats, idx, title):
    """Random subset of the inlier matches (green) and of the matches rejected
    by the outlier filtering (red) for one pair."""
    left_rgb, _, gray_r = images[idx]
    s = stats[idx]
    right_rgb = cv2.cvtColor(gray_r, cv2.COLOR_GRAY2RGB)
    flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    inlier_set = set(id(m) for m in s['inlier_matches'])
    outliers = [m for m in s['matches'] if id(m) not in inlier_set]
    # thousands of lines are unreadable, so draw a random subset of each
    draw_in = [s['inlier_matches'][k] for k in subsample(len(s['inlier_matches']), N_DRAW_MATCHES)]
    draw_out = [outliers[k] for k in subsample(len(outliers), N_DRAW_MATCHES)]
    inl_img = cv2.drawMatches(left_rgb, s['kp_l'], right_rgb, s['kp_r'], draw_in, None,
                              matchColor=(0, 255, 0), flags=flags)
    out_img = cv2.drawMatches(left_rgb, s['kp_l'], right_rgb, s['kp_r'], draw_out, None,
                              matchColor=(255, 0, 0), flags=flags)
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    ax[0].imshow(inl_img)
    ax[0].set_title('%s - pair %d: %d inliers of %d candidate matches (%d drawn)'
                    % (title, idx, len(s['inlier_matches']), len(s['matches']), len(draw_in)))
    ax[1].imshow(out_img)
    ax[1].set_title('%s - pair %d: %d rejected by outlier filtering (%d drawn)'
                    % (title, idx, len(outliers), len(draw_out)))
    for a in ax:
        a.set_axis_off()
    fig.tight_layout()


def plot_point_cloud(points, colours, poses, terrain, title):
    """3D point cloud (true colour), camera trajectory and reference terrain.
    The view is limited to the survey area (terrain extent, from just above the
    cameras to below the deepest terrain) so that gross outliers do not squash
    the plot; the number of points outside this region is reported."""
    cams = np.array([camera_to_world(np.zeros((1, 3)), poses['R'][:, :, i], poses['t'][:, i])[0]
                     for i in range(poses['R'].shape[2])])
    xlim = (np.min(terrain['X']), np.max(terrain['X']))
    ylim = (np.min(terrain['Y']), np.max(terrain['Y']))
    zlim = (cams[:, 2].min() - 0.5, np.nanmax(-terrain['height_grid']) + 1.0)
    in_view = ((points[:, 0] >= xlim[0]) & (points[:, 0] <= xlim[1]) &
               (points[:, 1] >= ylim[0]) & (points[:, 1] <= ylim[1]) &
               (points[:, 2] >= zlim[0]) & (points[:, 2] <= zlim[1]))
    n_out = int((~in_view).sum())
    points, colours = points[in_view], colours[in_view]
    idx = subsample(len(points), 60000)

    fig = plt.figure(figsize=(14, 6))
    # (a) reconstructed point cloud with image colours
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(points[idx, 0], points[idx, 1], points[idx, 2], c=colours[idx], s=1)
    ax.plot(cams[:, 0], cams[:, 1], cams[:, 2], 'r.-', label='left camera positions')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z (down) [m]')
    ax.invert_zaxis()
    ax.set_title('%s - point cloud (%d points, %d outside view)' % (title, len(points) + n_out, n_out))
    ax.legend()

    # (b) point cloud (coloured by depth) on top of the reference terrain
    # (terrain plotting code as given in the assignment)
    ax = fig.add_subplot(122, projection='3d')
    X, Y = np.meshgrid(np.flip(terrain['X']), terrain['Y'])
    ax.plot_surface(X, Y, -terrain['height_grid'], rstride=5, cstride=5, color='lightgray', alpha=0.5)
    ax.scatter(points[idx, 0], points[idx, 1], points[idx, 2], c=points[idx, 2], cmap='viridis', s=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z (down) [m]')
    ax.invert_zaxis()
    ax.set_title('Point cloud vs. reference terrain')
    fig.tight_layout()


def plot_terrain_comparison(points, err, terrain_model, title):
    """Footprint on the reference terrain, spatial error map and histogram."""
    valid = ~np.isnan(err)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))

    # (a) reference terrain with reconstructed footprint
    extent = [terrain_model.x[0], terrain_model.x[-1], terrain_model.y[0], terrain_model.y[-1]]
    im = ax[0].imshow(terrain_model.z, origin='lower', extent=extent, cmap='viridis', aspect='equal')
    idx = subsample(len(points), 20000)
    ax[0].scatter(points[idx, 0], points[idx, 1], s=0.2, c='r', alpha=0.3)
    ax[0].set_title('Reference terrain depth + point footprint (red)')
    ax[0].set_xlabel('X [m]')
    ax[0].set_ylabel('Y [m]')
    fig.colorbar(im, ax=ax[0], label='Z (down) [m]')

    # (b) spatial map of the height error
    idx = np.where(valid)[0]
    idx = idx[subsample(len(idx), 40000)]
    sc = ax[1].scatter(points[idx, 0], points[idx, 1], c=err[idx], s=1, cmap='coolwarm', vmin=-0.15, vmax=0.15)
    ax[1].set_aspect('equal')
    ax[1].set_title('Height error: point Z - reference Z')
    ax[1].set_xlabel('X [m]')
    ax[1].set_ylabel('Y [m]')
    fig.colorbar(sc, ax=ax[1], label='error [m]')

    # (c) histogram of errors
    e = err[valid]
    ax[2].hist(np.clip(e, -0.3, 0.3), bins=120, color='steelblue')
    ax[2].axvline(np.median(e), color='r', label='median %.3f m' % np.median(e))
    ax[2].set_xlabel('height error [m] (clipped to +-0.3)')
    ax[2].set_ylabel('number of points')
    ax[2].set_title('%s error distribution (RMSE %.3f m)' % (title, np.sqrt(np.mean(e ** 2))))
    ax[2].legend()
    fig.tight_layout()


def plot_comparison(results, title):
    """Bar charts summarising a set of pipeline configurations."""
    labels = [r['label'] for r in results]
    x = np.arange(len(labels))
    kp = [np.mean([s['n_kp_left'] for s in r['stats']]) for r in results]
    mt = [np.mean([s['n_matches'] for s in r['stats']]) for r in results]
    inl = [np.mean([len(s['points_cam']) for s in r['stats']]) for r in results]
    rate = [100.0 * i / m if m > 0 else 0 for i, m in zip(inl, mt)]
    tm = [np.mean([s['time'] for s in r['stats']]) for r in results]
    rmse = [r['summary']['rmse'] for r in results]
    mad = [r['summary']['mad'] for r in results]
    w5 = [100 * r['summary']['within_5cm'] for r in results]
    cov = [100.0 * r['summary']['occupied_cells_5cm'] / r['summary']['reference_cells_5cm'] for r in results]

    fig, ax = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(title)
    ax = ax.ravel()
    w = 0.27
    ax[0].bar(x - w, kp, w, label='keypoints (left)')
    ax[0].bar(x, mt, w, label='candidate matches')
    ax[0].bar(x + w, inl, w, label='final inliers')
    ax[0].set_title('Mean per stereo pair')
    ax[0].legend(fontsize=8)
    ax[1].bar(x, rate, color='tab:green')
    ax[1].set_title('Inlier rate (% of candidate matches)')
    ax[2].bar(x, tm, color='tab:gray')
    ax[2].set_title('Processing time per pair [s]')
    ax[3].bar(x, [r['summary']['n_points'] for r in results], color='tab:orange')
    ax[3].set_title('Total points in merged cloud')
    ax[4].bar(x, rmse, color='tab:blue')
    ax[4].set_title('RMSE vs. reference terrain [m]')
    ax[5].bar(x, mad, color='tab:cyan')
    ax[5].set_title('MAD (robust spread) [m]')
    ax[6].bar(x, w5, color='tab:red')
    ax[6].set_title('% of points within 5 cm of terrain')
    ax[7].bar(x, cov, color='tab:purple')
    ax[7].set_title('Coverage: % of 5 cm terrain cells with a point')
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    fig.tight_layout()


def plot_depth_error_by_range(results):
    """Median height error vs. distance to camera for each configuration:
    stereo depth uncertainty grows with Z^2 / (f * baseline)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        d = r['depth']
        e = np.abs(r['err'])
        ok = ~np.isnan(e)
        bins = np.linspace(1.0, 3.5, 11)
        centres, med = [], []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            sel = ok & (d >= b0) & (d < b1)
            if sel.sum() > 500:   # ignore sparsely populated bins
                centres.append(0.5 * (b0 + b1))
                med.append(np.median(e[sel]))
        ax.plot(centres, med, 'o-', label=r['label'])
    ax.set_xlabel('distance from left camera Z_cam [m]')
    ax.set_ylabel('median |height error| [m]')
    ax.set_title('Reconstruction error vs. range')
    ax.legend(fontsize=8)
    fig.tight_layout()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Further analysis (printed to the terminal)
# ---------------------------------------------------------------------------
SLOPE_CLASSES = ('flat', 'moderate', 'steep')


def rejection_stages(cfg, calib, poses, images, cache, terrain_model):
    """Effect of each outlier-rejection stage. For every pair, all candidate
    matches (nearest neighbour of every left descriptor) are triangulated and
    the stages are applied one at a time: ratio test -> epipolar check ->
    reprojection check. Each cumulative stage is transformed to
    the world frame and compared with the reference terrain.
    Returns a list of (stage name, n removed, summary, worst |error|)."""
    Pleft, Pright = projection_matrices(calib)
    F_cal = calibrated_fundamental_matrix(calib)
    features, _ = cache.get(images, cfg)
    names = ['candidate matches', '+ ratio test', '+ epipolar check',
             '+ reprojection check (final)']
    stage_pts = [[] for _ in names]
    for i, (kp_l, des_l, kp_r, des_r, norm) in enumerate(features):
        knn = [p for p in cv2.BFMatcher(norm).knnMatch(des_l, des_r, k=2) if len(p) == 2]
        best = [p[0] for p in knn]
        ratio_ok = np.array([p[0].distance < cfg['ratio'] * p[1].distance for p in knn])
        pts1 = np.float32([kp_l[m.queryIdx].pt for m in best])
        pts2 = np.float32([kp_r[m.trainIdx].pt for m in best])
        und1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), calib['Kl'], calib['Dl'], None, calib['Kl']).reshape(-1, 2)
        und2 = cv2.undistortPoints(pts2.reshape(-1, 1, 2), calib['Kr'], calib['Dr'], None, calib['Kr']).reshape(-1, 2)
        X = triangulate(Pleft, Pright, und1, und2)
        epi_ok = sampson_distance(F_cal, und1, und2) < EPIPOLAR_THRESH
        err = np.maximum(reprojection_error(Pleft, X, und1), reprojection_error(Pright, X, und2))
        rep_ok = err < REPROJ_THRESH
        masks = [np.ones(len(best), dtype=bool), ratio_ok, ratio_ok & epi_ok,
                 ratio_ok & epi_ok & rep_ok]
        Xw = camera_to_world(X, poses['R'][:, :, i], poses['t'][:, i])
        for k, m in enumerate(masks):
            stage_pts[k].append(Xw[m])
    rows, prev = [], None
    for name, pts in zip(names, stage_pts):
        pts = np.vstack(pts)
        e, s = compare_to_terrain(pts, terrain_model)
        worst = float(np.nanmax(np.abs(e)))
        rows.append((name, 0 if prev is None else prev - len(pts), s, worst))
        prev = len(pts)
    return rows


def print_rejection_stages(rows):
    print('\nOutlier rejection, stage by stage (all 49 pairs):')
    print('%-24s %9s %9s %9s %10s %9s %9s' % ('stage', 'points', 'removed', 'RMSE[m]',
                                             'med|e|[cm]', '<5cm', 'worst[m]'))
    for name, removed, s, worst in rows:
        print('%-24s %9d %9d %9.3f %10.1f %8.1f%% %9.1f'
              % (name, s['n_points'], removed, s['rmse'], 100 * s['median_abs'],
                 100 * s['within_5cm'], worst))


def terrain_slope_classes(terrain_model, cell=5):
    """Split the reference terrain into 5 cm cells and classify each cell by
    local slope into thirds: flat (mostly sand/rubble), moderate and steep
    (coral faces and edges). Returns (class index per cell, -1 = no data;
    slope edges in degrees)."""
    ny, nx = (terrain_model.z.shape[0] // cell) * cell, (terrain_model.z.shape[1] // cell) * cell
    blocks = terrain_model.z[:ny, :nx].reshape(ny // cell, cell, nx // cell, cell)
    with warnings.catch_warnings():                      # cells without terrain data -> NaN
        warnings.simplefilter('ignore', category=RuntimeWarning)
        zm = np.nanmean(blocks, axis=(1, 3))            # mean height of each cell (NaN = no data)
        gy, gx = np.gradient(zm, cell * 0.01)            # cell spacing in metres
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    ok = ~np.isnan(slope)
    edges = np.percentile(slope[ok], [100 / 3.0, 200 / 3.0])
    cls = np.full(slope.shape, -1)
    cls[ok] = np.digitize(slope[ok], edges)
    return cls, edges


def structure_coverage(results, terrain_model, cell=5):
    """Coverage and median |height error| on flat, moderate and steep terrain
    for each configuration. Coverage is measured only inside the area the
    cameras imaged (cells reconstructed by at least one configuration), so it
    shows which structures each detector misses."""
    cls, edges = terrain_slope_classes(terrain_model, cell)
    occ, idx = [], []
    for r in results:
        ref, i, j, _ = terrain_model.lookup(r['points'][:, 0], r['points'][:, 1])
        ok = ~np.isnan(ref)
        o = np.zeros(cls.shape, dtype=bool)
        ci, cj = i[ok] // cell, j[ok] // cell
        inside = (ci < cls.shape[0]) & (cj < cls.shape[1])
        o[ci[inside], cj[inside]] = True
        occ.append(o)
        idx.append((ok, ci, cj, inside))
    seen = np.any(occ, axis=0) & (cls >= 0)
    table = []
    for r, o, (ok, ci, cj, inside) in zip(results, occ, idx):
        e = np.abs(r['err'][ok])[inside]
        pc = cls[ci[inside], cj[inside]]
        cov = [100.0 * np.mean(o[seen & (cls == k)]) for k in range(3)]
        med = [100.0 * np.median(e[pc == k]) if np.any(pc == k) else np.nan for k in range(3)]
        table.append((r['label'], cov, med))
    n_cells = [int(np.sum(seen & (cls == k))) for k in range(3)]
    return table, edges, n_cells


def print_structure_coverage(table, edges, n_cells):
    print('\nTypes of structures covered (terrain split by slope, inside the imaged area):')
    print('  flat < %.0f deg (%d cells), moderate %.0f-%.0f deg (%d cells), steep > %.0f deg (%d cells)'
          % (edges[0], n_cells[0], edges[0], edges[1], n_cells[1], edges[1], n_cells[2]))
    print('%-22s %8s %8s %8s | %8s %8s %8s' % ('', 'cov flat', 'moderate', 'steep',
                                              'err flat', 'moderate', 'steep'))
    for label, cov, med in table:
        print('%-22s %7.1f%% %7.1f%% %7.1f%% | %6.1fcm %6.1fcm %6.1fcm' % ((label,) + tuple(cov) + tuple(med)))


def plot_structure_coverage(table, edges):
    """Bar charts of coverage and median error per slope class."""
    labels = [t[0] for t in table]
    x = np.arange(len(labels))
    w = 0.27
    names = ['flat (< %.0f deg)' % edges[0], 'moderate', 'steep (> %.0f deg)' % edges[1]]
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    for k in range(3):
        ax[0].bar(x + (k - 1) * w, [t[1][k] for t in table], w, label=names[k])
        ax[1].bar(x + (k - 1) * w, [t[2][k] for t in table], w, label=names[k])
    ax[0].set_title('Coverage of the imaged area by terrain slope [%]')
    ax[1].set_title('Median |height error| by terrain slope [cm]')
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
        a.legend(fontsize=8)
    fig.suptitle('Types of structures covered: flat sand/rubble vs steep coral')
    fig.tight_layout()


def brightness_split(results, images):
    """Number of points coming from dark vs bright image regions, for each
    configuration. Local brightness = 31x31 mean of the raw left grayscale
    image; the dark/bright threshold is its median over all images."""
    local = [cv2.blur(gray_l, (31, 31)) for _, gray_l, _ in images]
    thr = np.median(np.concatenate([l.ravel()[::50] for l in local]))
    rows = []
    for r in results:
        b = np.concatenate([local[i][np.round(s['pix_left'][:, 1]).astype(int),
                                     np.round(s['pix_left'][:, 0]).astype(int)]
                            for i, s in enumerate(r['stats'])])
        rows.append((r['label'], int(np.sum(b < thr)), int(np.sum(b >= thr))))
    return rows


def print_brightness_split(rows):
    print('\nPoints from dark vs bright image regions (split at the median local brightness):')
    for label, dark, bright in rows:
        print('%-22s dark %7d   bright %7d' % (label, dark, bright))
    if len(rows) == 2:
        print('%-22s dark x%.1f   bright x%.1f' % ('ratio', rows[0][1] / max(rows[1][1], 1),
                                                    rows[0][2] / max(rows[1][2], 1)))


def show_figures(saved):
    """Save every new figure to FIGURE_DIR, then draw all figures so they
    appear while the rest of the script is still running. Figures are saved
    immediately, so closing a window early does not lose the figure.
    saved = set of figure numbers already saved (updated in place)."""
    for n in plt.get_fignums():
        if n not in saved:
            plt.figure(n).savefig(os.path.join(FIGURE_DIR, 'q1_fig%02d.png' % n), dpi=150)
            saved.add(n)
    plt.pause(0.1)


def run_config(label, cfg, calib, poses, images, cache, terrain_model):
    """Build the point cloud for one configuration and evaluate it."""
    print('Running %s ...' % label)
    pts, cols, st = build_point_cloud(cfg, calib, poses, images, cache)
    e, s = compare_to_terrain(pts, terrain_model)
    depth = np.concatenate([x['points_cam'][:, 2] for x in st])
    return {'label': label, 'points': pts, 'colours': cols, 'stats': st,
            'summary': s, 'err': e, 'depth': depth}


def main():
    calib = load_pickle(calib_file)
    poses = load_pickle(pose_file)
    terrain = load_pickle(terrain_file)
    terrain_model = TerrainModel(terrain)
    n_pairs = poses['R'].shape[2]
    print('Stereo baseline: %.3f m, focal length: %.0f px'
          % (np.linalg.norm(calib['t']), calib['Kl'][0, 0]))

    # Load every stereo pair once (re-used by all configurations)
    print('Loading %d stereo pairs ...' % n_pairs)
    images = [load_stereo_pair(i, poses) for i in range(n_pairs)]
    cache = FeatureCache()
    plt.ion()   # interactive mode: figures are displayed as soon as they are drawn

    # figures are saved here for the report; remove figures from older runs
    os.makedirs(FIGURE_DIR, exist_ok=True)
    for f in os.listdir(FIGURE_DIR):
        if f.startswith('q1_fig') and f.endswith('.png'):
            os.remove(os.path.join(FIGURE_DIR, f))
    saved = set()

    # ---------------- main reconstruction ----------------
    main_res = run_config('SIFT (CONFIG)', CONFIG, calib, poses, images, cache, terrain_model)
    plot_keypoints(images, main_res['stats'], EXAMPLE_PAIR, 'SIFT')
    plot_example_matches(images, main_res['stats'], EXAMPLE_PAIR, 'SIFT')
    plot_point_cloud(main_res['points'], main_res['colours'], poses, terrain, 'SIFT')
    plot_terrain_comparison(main_res['points'], main_res['err'], terrain_model, 'SIFT')
    show_figures(saved)

    # ---------------- before vs after outlier rejection ----------------
    # (run straight after the main reconstruction so the cached SIFT features are reused)
    print_rejection_stages(rejection_stages(CONFIG, calib, poses, images, cache, terrain_model))

    # ---------------- detector / parameter comparison ----------------
    print('\nDetector / parameter comparison')
    results = [main_res]
    for label, cfg in COMPARISON_CONFIGS:
        results.append(run_config(label, cfg, calib, poses, images, cache, terrain_model))

    print('\nResults:')
    for r in results:
        print_summary(r['label'], r['summary'])

    plot_comparison(results, 'Detector and parameter comparison')
    plot_depth_error_by_range(results)
    for r in results:
        if r['label'] in ('SIFT no ratio test', 'ORB 10000', 'AKAZE default'):
            plot_example_matches(images, r['stats'], EXAMPLE_PAIR, r['label'])

    # ---------------- types of structures covered ----------------
    structure = [r for r in results if r['label'] in
                 ('SIFT (CONFIG)', 'SIFT no CLAHE', 'ORB 1000', 'ORB 10000', 'AKAZE default')]
    table, edges, n_cells = structure_coverage(structure, terrain_model)
    print_structure_coverage(table, edges, n_cells)
    plot_structure_coverage(table, edges)

    # ---------------- effect of CLAHE on dark / bright regions ----------------
    print_brightness_split(brightness_split(
        [r for r in results if r['label'] in ('SIFT (CONFIG)', 'SIFT no CLAHE')], images))
    show_figures(saved)

    print('Figures saved to %s' % FIGURE_DIR)
    plt.ioff()
    plt.show()  # keep all figure windows open until they are closed


if __name__ == '__main__':
    main()
