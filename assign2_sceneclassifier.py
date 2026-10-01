"""
AMME5710 Assignment 2 - Question 2
Scene classifier: feature extraction and the assign2_sceneclassifier function.

    class_labels = assign2_sceneclassifier(image_path)

returns the 7 scene labels (folder names of the dataset) ordered from the most
to the least likely class. The trained model is loaded from MODEL_FILE, which
sits next to this file and is produced by MainQ2.py.

The same feature extraction functions are used by MainQ2.py for training, so the
features seen at training and prediction time are always identical.

Each image is described by five feature blocks:
    colour  - colour histogram (global colour distribution)
    layout  - mean / std colour on a coarse grid (where the colours are, e.g. sky on top)
    hog     - histogram of oriented gradients (edge directions and their layout)
    lbp     - local binary pattern histograms (fine texture)
    gist    - Gabor filter energies on a grid (texture scale / orientation layout)

Dependencies: numpy, opencv-python, scikit-learn
"""

import os
import pickle

import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Trained model file (created by MainQ2.py), stored in the same folder as this file
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assign2_scene_model.pkl')

# ---------------------------------------------------------------------------
# Feature extraction settings (MainQ2.py varies some of these in its experiments;
# the settings used for training are saved in the model file)
# ---------------------------------------------------------------------------
#   colour_image_size   colour features are computed on an image of this size [px]
#   gray_image_size     texture features are computed on a grayscale image of this size [px]
#   clahe               True -> CLAHE on the grayscale image before texture features
#   colour_equalise     True -> histogram-equalise the brightness (V of HSV) of the colour image
#   colour_space        'RGB' | 'HSV' | 'Lab' for the colour histogram
#   hist_bins           bins per channel of the joint colour histogram
#   layout_grid         layout features on a layout_grid x layout_grid grid
#   hog_cell            HOG cell size [px] (block = 2x2 cells, stride = 1 cell)
#   lbp_radii           LBP neighbourhood radii [px]
#   gabor_wavelengths   Gabor filter wavelengths [px] (one scale each)
#   gabor_orientations  number of Gabor orientations per scale
#   gabor_grid          Gabor energies averaged on a gabor_grid x gabor_grid grid
FEATURE_PARAMS = {
    'colour_image_size': 128,
    'gray_image_size': 128,
    'clahe': False,
    'colour_equalise': False,
    'colour_space': 'HSV',
    'hist_bins': (8, 4, 4),
    'layout_grid': 4,
    'hog_cell': 32,
    'lbp_radii': (1, 2, 3),
    'gabor_wavelengths': (4, 8, 16, 32),
    'gabor_orientations': 6,
    'gabor_grid': 4,
}

BLOCKS = ('colour', 'layout', 'hog', 'lbp', 'gist')

# channel ranges of 8-bit OpenCV images, used for the colour histogram
_CHANNEL_RANGES = {'RGB': [0, 256, 0, 256, 0, 256],
                   'HSV': [0, 180, 0, 256, 0, 256],   # OpenCV hue is 0-179
                   'Lab': [0, 256, 0, 256, 0, 256]}
_COLOUR_CONVERSIONS = {'RGB': cv2.COLOR_BGR2RGB, 'HSV': cv2.COLOR_BGR2HSV, 'Lab': cv2.COLOR_BGR2Lab}


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------
def preprocess(img_bgr, params):
    """Resize the image and build the two inputs of the feature extractors:
    a small colour image (BGR), optionally with its brightness histogram-
    equalised, and a grayscale image, optionally CLAHE-equalised.
    All dataset images are 256x256; resizing also makes other image sizes work."""
    s = params['colour_image_size']
    colour = cv2.resize(img_bgr, (s, s), interpolation=cv2.INTER_AREA)
    if params.get('colour_equalise', False):
        # global histogram equalisation of the brightness only (V of HSV), so the
        # hue is unchanged: removes differences in exposure between photos
        hsv = cv2.cvtColor(colour, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        colour = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    g = params['gray_image_size']
    gray = cv2.cvtColor(cv2.resize(img_bgr, (g, g), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    if params['clahe']:
        # local contrast equalisation: texture is measured the same way in
        # bright (snow, sky) and dark (shadowed street) images
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    return colour, gray


def grid_mean(values, n):
    """Average an HxW(xC) array over an n x n grid of cells -> (n, n, C) array."""
    h, w = values.shape[:2]
    values = values.reshape(h, w, -1)
    ys = np.linspace(0, h, n + 1).astype(int)
    xs = np.linspace(0, w, n + 1).astype(int)
    out = np.zeros((n, n, values.shape[2]))
    for i in range(n):
        for j in range(n):
            out[i, j] = values[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean(axis=(0, 1))
    return out


# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------
def colour_histogram(colour, params):
    """Joint 3D colour histogram in the chosen colour space, normalised to sum 1.
    The square root is taken (Hellinger mapping) so that one dominant colour
    (e.g. a sky covering half the image) does not swamp the smaller bins."""
    space = params['colour_space']
    img = cv2.cvtColor(colour, _COLOUR_CONVERSIONS[space])
    hist = cv2.calcHist([img], [0, 1, 2], None, list(params['hist_bins']), _CHANNEL_RANGES[space])
    hist = hist.ravel() / hist.sum()
    return np.sqrt(hist)


def spatial_layout(colour, params):
    """Mean and standard deviation of the L, a, b channels in each cell of a
    coarse grid: describes where colours are (sky at the top, grass or snow at
    the bottom, ...). Lab is used because distances in Lab roughly follow
    perceived colour differences. Values are scaled to 0-1."""
    lab = cv2.cvtColor(colour, cv2.COLOR_BGR2Lab).astype(np.float64) / 255.0
    n = params['layout_grid']
    mean = grid_mean(lab, n)
    std = np.sqrt(np.maximum(grid_mean(lab ** 2, n) - mean ** 2, 0))
    return np.concatenate([mean.ravel(), std.ravel()])


def hog_features(gray, params):
    """Histogram of oriented gradients over the whole image (Week 7 tutorial,
    cv2.HOGDescriptor). Large cells capture the dominant edge directions of the
    scene layout: verticals of buildings, converging road edges, horizon line."""
    size = gray.shape[0]
    c = params['hog_cell']
    hog = cv2.HOGDescriptor(_winSize=(size, size), _blockSize=(2 * c, 2 * c),
                            _blockStride=(c, c), _cellSize=(c, c), _nbins=9)
    return hog.compute(gray).ravel()


def lbp_riu2(gray, radius):
    """Rotation-invariant uniform local binary pattern (8 neighbours) of every
    pixel. Each neighbour at distance `radius` is compared with the centre pixel;
    a pattern with at most 2 bit transitions around the circle (a 'uniform' edge,
    corner, spot or flat patch) is coded by its number of set bits (0-8), every
    other pattern gets code 9. Returns an integer image of codes 0-9."""
    p = 8
    padded = np.pad(gray.astype(np.int16), radius, mode='edge')
    h, w = gray.shape
    centre = gray.astype(np.int16)
    bits = []
    for k in range(p):
        a = 2 * np.pi * k / p
        dy, dx = int(round(-radius * np.sin(a))), int(round(radius * np.cos(a)))
        neighbour = padded[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
        bits.append(neighbour >= centre)
    bits = np.array(bits, dtype=np.int16)
    transitions = np.abs(bits - np.roll(bits, 1, axis=0)).sum(axis=0)
    return np.where(transitions <= 2, bits.sum(axis=0), p + 1)


def lbp_histogram(gray, params):
    """Normalised histograms of LBP codes at several radii (fine to medium texture:
    smooth sand/sky/snow vs. grass, foliage, balls, windows)."""
    feats = []
    for r in params['lbp_radii']:
        codes = lbp_riu2(gray, r)
        hist = np.bincount(codes.ravel(), minlength=10).astype(np.float64)
        feats.append(hist / hist.sum())
    return np.concatenate(feats)


_GABOR_CACHE = {}


def gabor_bank(wavelengths, n_orient):
    """Pairs of even/odd (cosine/sine) Gabor kernels for every scale and orientation."""
    key = (tuple(wavelengths), n_orient)
    if key not in _GABOR_CACHE:
        bank = []
        for lam in wavelengths:
            sigma = 0.56 * lam                       # ~1 octave bandwidth
            ksize = int(2 * np.ceil(2.5 * sigma) + 1)
            for k in range(n_orient):
                theta = np.pi * k / n_orient
                even = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, 0.5, 0, ktype=cv2.CV_32F)
                odd = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, 0.5, np.pi / 2, ktype=cv2.CV_32F)
                even -= even.mean()                  # no response to flat regions
                bank.append((even, odd))
        _GABOR_CACHE[key] = bank
    return _GABOR_CACHE[key]


def gabor_energy_maps(gray, params):
    """Magnitude of the response of each Gabor filter -> (H, W, n_filters)."""
    img = gray.astype(np.float32) / 255.0
    maps = []
    for even, odd in gabor_bank(params['gabor_wavelengths'], params['gabor_orientations']):
        re = cv2.filter2D(img, cv2.CV_32F, even, borderType=cv2.BORDER_REFLECT)
        im = cv2.filter2D(img, cv2.CV_32F, odd, borderType=cv2.BORDER_REFLECT)
        maps.append(np.sqrt(re ** 2 + im ** 2))
    return np.stack(maps, axis=2)


def gist_features(gray, params):
    """GIST-style descriptor (Oliva & Torralba): Gabor energy at each scale and
    orientation, averaged over a coarse grid. Captures the 'spatial envelope' of
    a scene - how much texture of each size and direction is in each region."""
    energy = gabor_energy_maps(gray, params)
    return grid_mean(energy, params['gabor_grid']).ravel()


_EXTRACTORS = {
    'colour': lambda colour, gray, p: colour_histogram(colour, p),
    'layout': lambda colour, gray, p: spatial_layout(colour, p),
    'hog': lambda colour, gray, p: hog_features(gray, p),
    'lbp': lambda colour, gray, p: lbp_histogram(gray, p),
    'gist': lambda colour, gray, p: gist_features(gray, p),
}


def extract_features(img_bgr, params=FEATURE_PARAMS, blocks=BLOCKS):
    """Compute the requested feature blocks of one image -> {block name: 1D vector}."""
    colour, gray = preprocess(img_bgr, params)
    return {b: _EXTRACTORS[b](colour, gray, params).astype(np.float64) for b in blocks}


def stack_blocks(features, blocks):
    """Concatenate feature blocks. `features` maps block name -> (N, d) array
    (or a 1D vector for one image). Returns the matrix and the size of each block."""
    parts = [np.atleast_2d(features[b]) for b in blocks]
    return np.hstack(parts), [p.shape[1] for p in parts]


# ---------------------------------------------------------------------------
# Scaling of concatenated feature blocks
# ---------------------------------------------------------------------------
class BlockScaler(BaseEstimator, TransformerMixin):
    """Standardise every feature (zero mean, unit variance over the training set),
    then divide each block by sqrt(block size). Every block then contributes the
    same total variance to the distance, so that a 384-value GIST block does not
    outweigh a 30-value LBP block in the KNN / SVM distance."""

    def __init__(self, block_sizes=None):
        self.block_sizes = block_sizes

    def fit(self, X, y=None):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0     # constant features
        sizes = self.block_sizes if self.block_sizes is not None else [X.shape[1]]
        self.weight_ = np.concatenate([np.full(s, 1.0 / np.sqrt(s)) for s in sizes])
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_ * self.weight_


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
def rank_classes(model, X):
    """Class labels of every row of X, ordered from most to least likely.
    Uses the class probabilities (predict_proba) of the trained model."""
    proba = model.predict_proba(X)
    order = np.argsort(-proba, axis=1, kind='stable')
    return np.asarray(model.classes_)[order]


_MODEL = None


def load_model(path=MODEL_FILE):
    """Load the trained model (once) from the pickle file written by MainQ2.py:
    a dict with the sklearn pipeline, the feature blocks and feature settings."""
    global _MODEL
    if _MODEL is None or _MODEL.get('path') != path:
        with open(path, 'rb') as f:
            _MODEL = pickle.load(f)
        _MODEL['path'] = path
    return _MODEL


def assign2_sceneclassifier(image_path):
    """Classify the scene in an RGB image file.
    Returns a list of all class labels (dataset folder names), ordered from the
    most likely to the least likely; class_labels[0] is the predicted scene."""
    m = load_model()
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)   # 3-channel BGR, also for grayscale files
    if img is None:
        raise IOError('Could not read image: %s' % image_path)
    feats = extract_features(img, m['params'], m['blocks'])
    x, _ = stack_blocks(feats, m['blocks'])
    return list(rank_classes(m['model'], x)[0])


if __name__ == '__main__':
    import sys
    for path in sys.argv[1:]:
        print(path, assign2_sceneclassifier(path))
