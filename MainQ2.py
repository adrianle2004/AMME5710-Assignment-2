"""
AMME5710 Assignment 2 - Question 2
Scene Classification

Follows the structure of the Week 7 tutorial (feature extraction -> scikit-learn
classifier -> performance metrics). The script:
  1. loads the 7-class places dataset and splits it into a training set and a
     held-out test set (stratified),
  2. compares pre-processing choices and feature parameters (repeated
     stratified 5-fold cross validation on the training set),
  3. compares feature sets and classifiers (KNN, SVM, random forest), with a
     cross-validated grid search of each classifier's parameters,
  4. selects the best model by cross-validation accuracy and evaluates it once
     on the test set (accuracy, precision, recall, F1, top-k, confusion matrix),
  5. retrains the selected model on all images and saves it to the model file
     used by assign2_sceneclassifier.py.

The feature extraction functions are in assign2_sceneclassifier.py.

Dependencies: numpy, opencv-python, scikit-learn, matplotlib
"""

import os
import pickle
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             top_k_accuracy_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict,
                                     learning_curve, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from assign2_sceneclassifier import (BLOCKS, FEATURE_PARAMS, MODEL_FILE, BlockScaler,
                                     assign2_sceneclassifier, extract_features, gabor_energy_maps,
                                     lbp_riu2, preprocess, rank_classes, spatial_layout, stack_blocks)

# ---------------------------------------------------------------------------
# Paths (change this to the location of the data on your computer)
# ---------------------------------------------------------------------------
dataset_dir = 'assignment2_places/assignment2_places'   # one sub-folder per class

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
TEST_FRACTION = 0.2      # fraction of each class held out as the test set
CV_FOLDS = 5             # stratified k-fold cross-validation on the training set ...
CV_REPEATS = 3           # ... repeated with different folds (lower noise in the mean score)
RANDOM_SEED = 0          # fixes the split, the folds and the random forest
FINAL_TRAIN_ON_ALL = True  # retrain the selected model on all images before saving
FIGURE_DIR = 'figures_q2'  # figures are also saved here for the report

# Feature sets compared (each is a list of feature blocks, see assign2_sceneclassifier.py)
FEATURE_SETS = [
    ('colour',                ['colour']),
    ('layout',                ['layout']),
    ('hog',                   ['hog']),
    ('lbp',                   ['lbp']),
    ('gist',                  ['gist']),
    ('colour+layout',         ['colour', 'layout']),
    ('hog+lbp+gist',          ['hog', 'lbp', 'gist']),
    ('all',                   list(BLOCKS)),
]

# Classifiers and the parameter grids searched by cross-validation.
# Features are scaled by BlockScaler, so each block has a total variance of 1 and
# squared distances are ~2 x (number of blocks): the RBF gamma grid covers that range.
CLASSIFIERS = [
    ('KNN', KNeighborsClassifier(),
     {'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21],
      'weights': ['uniform', 'distance'],
      'metric': ['euclidean', 'manhattan']}),
    ('SVM', SVC(),
     [{'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]},
      {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.01, 0.03, 0.1, 0.3, 1, 3]}]),
    ('Random forest', RandomForestClassifier(n_estimators=500, random_state=RANDOM_SEED),
     {'max_features': ['sqrt', 0.2]}),
]

# Pre-processing / feature-parameter variants: (label, changes to FEATURE_PARAMS,
# feature blocks used). Each is scored with an RBF SVM by cross-validation.
PREPROC_EXPERIMENTS = [
    ('colour hist RGB',       {'colour_space': 'RGB', 'hist_bins': (4, 4, 4)}, ['colour']),
    ('colour hist HSV',       {}, ['colour']),
    ('colour hist Lab',       {'colour_space': 'Lab', 'hist_bins': (4, 4, 4)}, ['colour']),
    ('HSV bins 4x2x2',        {'hist_bins': (4, 2, 2)}, ['colour']),       # colour quantisation
    ('HSV bins 16x4x4',       {'hist_bins': (16, 4, 4)}, ['colour']),
    ('HSV bins 8x8x8',        {'hist_bins': (8, 8, 8)}, ['colour']),
    ('colour+layout',         {}, ['colour', 'layout']),
    ('colour+layout equalised', {'colour_equalise': True}, ['colour', 'layout']),
    ('layout grid 2x2',       {'layout_grid': 2}, ['layout']),
    ('layout grid 4x4',       {}, ['layout']),
    ('texture no CLAHE',      {}, ['hog', 'lbp', 'gist']),
    ('texture with CLAHE',    {'clahe': True}, ['hog', 'lbp', 'gist']),
    ('HOG cell 16',           {'hog_cell': 16}, ['hog']),
    ('HOG cell 32',           {}, ['hog']),
    ('GIST grid 2x2',         {'gabor_grid': 2}, ['gist']),
    ('GIST grid 4x4',         {}, ['gist']),
]
PREPROC_SVM_GRID = {'kernel': ['rbf'], 'C': [1, 10, 100], 'gamma': [0.1, 0.3, 1]}

# Feature sets compared class by class in the cross-validated confusion analysis
CLASS_ANALYSIS_SETS = ['colour', 'layout', 'hog+lbp+gist', 'colour+layout', 'all']

N_EXAMPLES_PER_CLASS = 5     # images per class in the dataset figure
FEATURE_EXAMPLES = ['urban', 'desert', 'ball_pit']   # classes shown in the feature figure


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(root):
    """Read every image of every class folder. The label of an image is the
    name of its folder. Returns the images (BGR), labels, paths and class names."""
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    images, labels, paths = [], [], []
    for c in classes:
        for f in sorted(os.listdir(os.path.join(root, c))):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                p = os.path.join(root, c, f)
                images.append(cv2.imread(p, cv2.IMREAD_COLOR))
                labels.append(c)
                paths.append(p)
    return images, np.array(labels), paths, classes


def extract_all(images, params, blocks):
    """Feature blocks for every image -> {block name: (N, d) array}."""
    feats = [extract_features(img, params, blocks) for img in images]
    return {b: np.array([f[b] for f in feats]) for b in blocks}


# ---------------------------------------------------------------------------
# Model selection and evaluation
# ---------------------------------------------------------------------------
def make_pipeline(block_sizes, clf):
    """Block-wise feature scaling followed by the classifier."""
    return Pipeline([('scale', BlockScaler(block_sizes)), ('clf', clf)])


def prefix_grid(grid):
    """Add the pipeline step name to the parameter names of a grid."""
    if isinstance(grid, list):
        return [prefix_grid(g) for g in grid]
    return {'clf__' + k: v for k, v in grid.items()}


def grid_search(X, y, block_sizes, clf, grid, cv):
    """Cross-validated grid search over the classifier parameters (accuracy)."""
    gs = GridSearchCV(make_pipeline(block_sizes, clone(clf)), prefix_grid(grid),
                      scoring='accuracy', cv=cv)
    gs.fit(X, y)
    return gs


def with_probabilities(model):
    """Copy of a pipeline whose classifier outputs class probabilities (needed to
    rank the classes). SVC computes them with Platt scaling."""
    model = clone(model)
    if isinstance(model.named_steps['clf'], SVC):
        model.set_params(clf__probability=True, clf__random_state=RANDOM_SEED)
    return model


def evaluate(model, X, y, classes):
    """Test-set metrics of a fitted model. The prediction is the first class of
    the probability ranking, i.e. exactly what assign2_sceneclassifier returns."""
    ranked = rank_classes(model, X)
    pred = ranked[:, 0]
    proba = model.predict_proba(X)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    return {
        'pred': pred,
        'accuracy': accuracy_score(y, pred),
        'precision': prec, 'recall': rec, 'f1': f1,
        'macro_f1': f1.mean(),
        'topk': [top_k_accuracy_score(y, proba, k=k, labels=model.classes_)
                 for k in range(1, len(classes) + 1)],
        'confusion': confusion_matrix(y, pred, labels=classes),
    }


def describe_params(params):
    """Short text of the best grid-search parameters."""
    return ', '.join('%s=%s' % (k.replace('clf__', ''), v) for k, v in sorted(params.items()))


def cv_class_analysis(feats, idx_train, y_train, classes, table, clf_name):
    """Which classes each feature type separates. For each feature set in
    CLASS_ANALYSIS_SETS, the best model of the selected classifier type predicts
    every training image once by 5-fold cross-validation (cross_val_predict).
    Returns per-class recall for each set, the confusion matrix of each set and
    the most-confused class pairs of each set."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    out = {}
    for fs_name in CLASS_ANALYSIS_SETS:
        entry = table[(fs_name, clf_name)]
        X, _ = stack_blocks(feats, entry['blocks'])
        pred = cross_val_predict(clone(entry['gs'].best_estimator_), X[idx_train], y_train, cv=cv)
        cm = confusion_matrix(y_train, pred, labels=classes)
        recall = cm.diagonal() / cm.sum(axis=1)
        pairs = [(cm[i, j] + cm[j, i], classes[i], classes[j], cm[i, j], cm[j, i])
                 for i in range(len(classes)) for j in range(i + 1, len(classes)) if cm[i, j] + cm[j, i] > 0]
        out[fs_name] = {'recall': recall, 'confusion': cm, 'accuracy': np.mean(pred == y_train),
                        'pairs': sorted(pairs, reverse=True)}
    return out


def print_class_analysis(analysis, classes, clf_name):
    print('\nPer-class recall by feature set (%s, 5-fold cross_val_predict on the training set):' % clf_name)
    print('  %-14s' % 'class' + ''.join('%15s' % f for f in CLASS_ANALYSIS_SETS))
    for i, c in enumerate(classes):
        print('  %-14s' % c + ''.join('%15.2f' % analysis[f]['recall'][i] for f in CLASS_ANALYSIS_SETS))
    print('  %-14s' % 'accuracy' + ''.join('%15.3f' % analysis[f]['accuracy'] for f in CLASS_ANALYSIS_SETS))
    for f in ('colour+layout', 'all'):
        print('  Most-confused pairs (%s): %s' % (f, '; '.join(
            '%s<->%s %d (%d/%d)' % (a, b, n, ab, ba) for n, a, b, ab, ba in analysis[f]['pairs'][:4])))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def plot_dataset_examples(images, labels, classes):
    """A few example images of every class."""
    n = N_EXAMPLES_PER_CLASS
    fig, ax = plt.subplots(len(classes), n, figsize=(1.8 * n, 1.8 * len(classes)))
    for i, c in enumerate(classes):
        idx = np.where(labels == c)[0][:n]
        for j, k in enumerate(idx):
            ax[i, j].imshow(rgb(images[k]))
            ax[i, j].set_xticks([]), ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(c, fontsize=11)
    fig.suptitle('Dataset: %d classes x %d images (examples)' % (len(classes), len(labels) // len(classes)))
    fig.tight_layout()


def plot_feature_examples(images, labels, feats):
    """What each feature block sees, for one example image of a few classes."""
    p = FEATURE_PARAMS
    fig, ax = plt.subplots(len(FEATURE_EXAMPLES), 5, figsize=(16, 3.3 * len(FEATURE_EXAMPLES)))
    for row, c in enumerate(FEATURE_EXAMPLES):
        k = np.where(labels == c)[0][0]
        colour, gray = preprocess(images[k], p)
        # layout: mean Lab colour of each grid cell, shown back in RGB
        n = p['layout_grid']
        mean_lab = spatial_layout(colour, p)[:n * n * 3].reshape(n, n, 3)
        layout_img = cv2.cvtColor(np.uint8(np.clip(mean_lab * 255, 0, 255)), cv2.COLOR_Lab2RGB)
        # gist: total Gabor energy of the finest two scales, summed over orientations
        energy = gabor_energy_maps(gray, p)[:, :, :2 * p['gabor_orientations']].sum(axis=2)
        ax[row, 0].imshow(rgb(images[k]))
        ax[row, 0].set_title('%s (input)' % c)
        ax[row, 1].bar(np.arange(len(feats['colour'][k])), feats['colour'][k], width=1.0, color='0.3')
        ax[row, 1].set_title('colour: HSV histogram (sqrt)')
        ax[row, 1].set_xlabel('bin (hue-major)')
        ax[row, 2].imshow(layout_img, interpolation='nearest')
        ax[row, 2].set_title('layout: mean colour, %dx%d grid' % (n, n))
        ax[row, 3].imshow(lbp_riu2(gray, 1), cmap='tab10', vmin=0, vmax=9)
        ax[row, 3].set_title('lbp: pattern codes (r=1)')
        ax[row, 4].imshow(energy, cmap='magma')
        ax[row, 4].set_title('gist: Gabor energy (fine scales)')
        for j in (0, 2, 3, 4):
            ax[row, j].set_axis_off()
    fig.tight_layout()


def plot_preproc_results(results):
    """Cross-validation accuracy of the pre-processing / feature-parameter variants."""
    labels = [r['label'] for r in results]
    means = [r['mean'] for r in results]
    stds = [r['std'] for r in results]
    default = [not r['changes'] for r in results]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colours = ['tab:blue' if d else 'tab:gray' for d in default]
    ax.barh(np.arange(len(labels)), means, xerr=stds, color=colours, capsize=3)
    for i, m in enumerate(means):
        ax.text(m + 0.01, i, '%.3f' % m, va='center', fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('%dx%d-fold CV accuracy (RBF SVM), mean +- std' % (CV_REPEATS, CV_FOLDS))
    ax.set_title('Pre-processing and feature parameters (blue = setting used)')
    fig.tight_layout()


def plot_model_comparison(table):
    """CV accuracy of every feature set with every classifier."""
    fs_names = [f for f, _ in FEATURE_SETS]
    clf_names = [c for c, _, _ in CLASSIFIERS]
    width = 0.8 / len(clf_names)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for j, c in enumerate(clf_names):
        means = [table[(f, c)]['mean'] for f in fs_names]
        stds = [table[(f, c)]['std'] for f in fs_names]
        x = np.arange(len(fs_names)) + (j - (len(clf_names) - 1) / 2) * width
        ax.bar(x, means, width, yerr=stds, capsize=2, label=c)
    ax.set_xticks(np.arange(len(fs_names)))
    ax.set_xticklabels(fs_names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('%dx%d-fold CV accuracy (best parameters)' % (CV_REPEATS, CV_FOLDS))
    ax.set_title('Feature sets and classifiers (training set, %d images)' % table['n_train'])
    ax.axhline(1.0 / 7, color='k', ls=':', lw=1, label='chance')
    ax.legend(loc='upper center', ncol=4)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()


def plot_parameter_selection(knn_gs, svm_gs, set_name):
    """Parameter selection curves for the chosen feature set: KNN accuracy vs k,
    RBF SVM accuracy over C and gamma, linear SVM accuracy vs C."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    # KNN: accuracy vs k, for each weighting, with the best distance metric
    r = knn_gs.cv_results_
    metric = knn_gs.best_params_['clf__metric']
    for w in ('uniform', 'distance'):
        sel = [i for i, p in enumerate(r['params'])
               if p['clf__weights'] == w and p['clf__metric'] == metric]
        ks = [r['params'][i]['clf__n_neighbors'] for i in sel]
        ax[0].errorbar(ks, r['mean_test_score'][sel], yerr=r['std_test_score'][sel],
                       marker='o', capsize=3, label='weights=%s' % w)
    ax[0].set_xlabel('k (number of neighbours)')
    ax[0].set_ylabel('CV accuracy')
    ax[0].set_title('KNN (%s, metric=%s)' % (set_name, metric))
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    # SVM RBF: C x gamma heat map
    r = svm_gs.cv_results_
    rbf = [(p['clf__C'], p['clf__gamma'], s) for p, s in zip(r['params'], r['mean_test_score'])
           if p['clf__kernel'] == 'rbf']
    Cs = sorted(set(c for c, _, _ in rbf))
    gammas = sorted(set(g for _, g, _ in rbf))
    grid = np.zeros((len(Cs), len(gammas)))
    for c, g, s in rbf:
        grid[Cs.index(c), gammas.index(g)] = s
    im = ax[1].imshow(grid, cmap='viridis', vmin=grid.min(), vmax=grid.max())
    for i in range(len(Cs)):
        for j in range(len(gammas)):
            ax[1].text(j, i, '%.2f' % grid[i, j], ha='center', va='center', fontsize=8,
                       color='w' if grid[i, j] < (grid.min() + grid.max()) / 2 else 'k')
    ax[1].set_xticks(range(len(gammas)))
    ax[1].set_xticklabels([str(g) for g in gammas])
    ax[1].set_yticks(range(len(Cs)))
    ax[1].set_yticklabels([str(c) for c in Cs])
    ax[1].set_xlabel('gamma')
    ax[1].set_ylabel('C')
    ax[1].set_title('RBF SVM CV accuracy (%s)' % set_name)
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    # SVM linear: accuracy vs C
    lin = [(p['clf__C'], s, sd) for p, s, sd in zip(r['params'], r['mean_test_score'], r['std_test_score'])
           if p['clf__kernel'] == 'linear']
    ax[2].errorbar([c for c, _, _ in lin], [s for _, s, _ in lin], yerr=[sd for _, _, sd in lin],
                   marker='o', capsize=3, label='linear SVM')
    ax[2].axhline(grid.max(), color='tab:red', ls='--', label='best RBF SVM')
    ax[2].set_xscale('log')
    ax[2].set_xlabel('C')
    ax[2].set_ylabel('CV accuracy')
    ax[2].set_title('Linear SVM (%s)' % set_name)
    ax[2].legend()
    ax[2].grid(alpha=0.3)
    fig.tight_layout()


def plot_learning_curve(model, X, y, cv, title):
    """CV accuracy vs number of training images, for the selected model."""
    sizes, train_sc, val_sc = learning_curve(clone(model), X, y, cv=cv,
                                             train_sizes=np.linspace(0.15, 1.0, 7), scoring='accuracy')
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for sc, lab in ((train_sc, 'training'), (val_sc, 'cross-validation')):
        ax.plot(sizes, sc.mean(axis=1), 'o-', label=lab)
        ax.fill_between(sizes, sc.mean(axis=1) - sc.std(axis=1), sc.mean(axis=1) + sc.std(axis=1), alpha=0.2)
    ax.set_xlabel('number of training images')
    ax.set_ylabel('accuracy')
    ax.set_title('Learning curve: %s' % title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return sizes, val_sc.mean(axis=1)


def plot_test_results(m, classes, title):
    """Confusion matrix and per-class precision / recall / F1 on the test set."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1, 1.2]})
    cm = m['confusion']
    ax[0].imshow(cm, cmap='Blues')
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax[0].text(j, i, cm[i, j], ha='center', va='center',
                           color='w' if cm[i, j] > cm.max() / 2 else 'k')
    ax[0].set_xticks(range(len(classes)))
    ax[0].set_xticklabels(classes, rotation=45, ha='right')
    ax[0].set_yticks(range(len(classes)))
    ax[0].set_yticklabels(classes)
    ax[0].set_xlabel('predicted')
    ax[0].set_ylabel('true')
    ax[0].set_title('Test confusion matrix - accuracy %.3f' % m['accuracy'])
    x = np.arange(len(classes))
    for k, (key, lab) in enumerate((('precision', 'precision'), ('recall', 'recall'), ('f1', 'F1'))):
        ax[1].bar(x + (k - 1) * 0.27, m[key], 0.27, label=lab)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(classes, rotation=45, ha='right')
    ax[1].set_ylim(0, 1.2)
    ax[1].set_title('Per-class test metrics - macro F1 %.3f' % m['macro_f1'])
    ax[1].legend(loc='upper center', ncol=3)
    ax[1].grid(axis='y', alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()


def plot_topk(test_results, n_classes):
    """Top-k test accuracy for the best model of each classifier type."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ks = np.arange(1, n_classes + 1)
    for name, m in test_results:
        ax.plot(ks, m['topk'], 'o-', label='%s (top-1 %.3f)' % (name, m['topk'][0]))
    ax.set_xlabel('k')
    ax.set_ylabel('top-k test accuracy')
    ax.set_title('Top-k accuracy (test set)')
    ax.set_ylim(min(m['topk'][0] for _, m in test_results) - 0.05, 1.01)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_class_analysis(analysis, classes, clf_name):
    """Cross-validated confusion matrix of the selected feature set and the
    per-class recall of each feature set."""
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={'width_ratios': [1, 1.6]})
    cm = analysis['all']['confusion']
    ax[0].imshow(cm, cmap='Blues')
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax[0].text(j, i, cm[i, j], ha='center', va='center', color='w' if cm[i, j] > cm.max() / 2 else 'k')
    ax[0].set_xticks(range(len(classes)))
    ax[0].set_xticklabels(classes, rotation=45, ha='right')
    ax[0].set_yticks(range(len(classes)))
    ax[0].set_yticklabels(classes)
    ax[0].set_xlabel('predicted')
    ax[0].set_ylabel('true')
    ax[0].set_title('CV confusion matrix, all features + %s\n(%d training images, each predicted once)'
                    % (clf_name, cm.sum()))
    x = np.arange(len(classes))
    w = 0.8 / len(CLASS_ANALYSIS_SETS)
    for k, f in enumerate(CLASS_ANALYSIS_SETS):
        ax[1].bar(x + (k - (len(CLASS_ANALYSIS_SETS) - 1) / 2) * w, analysis[f]['recall'], w, label=f)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(classes, rotation=45, ha='right')
    ax[1].set_ylim(0, 1.15)
    ax[1].set_ylabel('CV recall')
    ax[1].set_title('Per-class recall by feature set (%s)' % clf_name)
    ax[1].legend(ncol=5, fontsize=8, loc='upper center')
    ax[1].grid(axis='y', alpha=0.3)
    fig.tight_layout()


def plot_misclassified(images, y_true, ranked, idx_test):
    """Every misclassified test image with its true label and top-2 predictions."""
    wrong = np.where(ranked[:, 0] != y_true)[0]
    if len(wrong) == 0:
        return
    cols = min(6, len(wrong))
    rows = int(np.ceil(len(wrong) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows), squeeze=False)
    for a in ax.ravel():
        a.set_axis_off()
    for a, w in zip(ax.ravel(), wrong):
        a.imshow(rgb(images[idx_test[w]]))
        a.set_title('true: %s\npred: %s, %s' % (y_true[w], ranked[w, 0], ranked[w, 1]), fontsize=9)
    fig.suptitle('Misclassified test images (%d of %d)' % (len(wrong), len(y_true)))
    fig.tight_layout()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def show_figures(saved):
    """Save every new figure to FIGURE_DIR, then draw all figures so they
    appear while the rest of the script is still running.
    saved = set of figure numbers already saved (updated in place)."""
    for n in plt.get_fignums():
        if n not in saved:
            plt.figure(n).savefig(os.path.join(FIGURE_DIR, 'q2_fig%02d.png' % n), dpi=150)
            saved.add(n)
    plt.pause(0.1)


def main():
    t0 = time.time()
    plt.ion()   # interactive mode: figures are displayed as soon as they are drawn
    os.makedirs(FIGURE_DIR, exist_ok=True)
    for f in os.listdir(FIGURE_DIR):   # remove figures from older runs
        if f.startswith('q2_fig') and f.endswith('.png'):
            os.remove(os.path.join(FIGURE_DIR, f))
    saved = set()

    # ---------------- data ----------------
    images, labels, paths, classes = load_dataset(dataset_dir)
    print('Loaded %d images, %d classes: %s' % (len(images), len(classes), ', '.join(classes)))
    idx_train, idx_test = train_test_split(np.arange(len(labels)), test_size=TEST_FRACTION,
                                           stratify=labels, random_state=RANDOM_SEED)
    y_train, y_test = labels[idx_train], labels[idx_test]
    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_SEED)
    print('Train %d / test %d images (stratified), %dx%d-fold CV on the training set'
          % (len(idx_train), len(idx_test), CV_REPEATS, CV_FOLDS))

    print('Extracting features ...')
    feats = extract_all(images, FEATURE_PARAMS, BLOCKS)
    for b in BLOCKS:
        print('  %-7s %4d values' % (b, feats[b].shape[1]))
    plot_dataset_examples(images, labels, classes)
    plot_feature_examples(images, labels, feats)
    show_figures(saved)

    # ---------------- pre-processing / feature parameters ----------------
    print('\nPre-processing and feature parameters (RBF SVM, %dx%d-fold CV accuracy)' % (CV_REPEATS, CV_FOLDS))
    svm = [c for n, c, _ in CLASSIFIERS if n == 'SVM'][0]
    preproc = []
    for label, changes, blocks in PREPROC_EXPERIMENTS:
        if changes:
            params = dict(FEATURE_PARAMS, **changes)
            f = extract_all(images, params, blocks)
        else:
            f = feats
        X, sizes = stack_blocks(f, blocks)
        gs = grid_search(X[idx_train], y_train, sizes, svm, PREPROC_SVM_GRID, cv)
        i = gs.best_index_
        preproc.append({'label': label, 'changes': changes, 'mean': gs.best_score_,
                        'std': gs.cv_results_['std_test_score'][i]})
        print('  %-20s %.3f +- %.3f' % (label, gs.best_score_, gs.cv_results_['std_test_score'][i]))
    plot_preproc_results(preproc)
    show_figures(saved)

    # ---------------- feature sets x classifiers ----------------
    print('\nFeature sets x classifiers (%dx%d-fold CV accuracy, best parameters)' % (CV_REPEATS, CV_FOLDS))
    table = {'n_train': len(idx_train)}
    for fs_name, blocks in FEATURE_SETS:
        X, sizes = stack_blocks(feats, blocks)
        for clf_name, clf, grid in CLASSIFIERS:
            gs = grid_search(X[idx_train], y_train, sizes, clf, grid, cv)
            i = gs.best_index_
            table[(fs_name, clf_name)] = {'gs': gs, 'mean': gs.best_score_,
                                          'std': gs.cv_results_['std_test_score'][i],
                                          'blocks': blocks}
            print('  %-14s %-14s %.3f +- %.3f  (%s)' % (fs_name, clf_name, gs.best_score_,
                                                       gs.cv_results_['std_test_score'][i],
                                                       describe_params(gs.best_params_)))
    plot_model_comparison(table)

    # model selection: the highest CV accuracy (the test set is not used)
    best_key = max((k for k in table if k != 'n_train'), key=lambda k: table[k]['mean'])
    best_set, best_clf = best_key
    best = table[best_key]
    print('\nSelected model: %s features + %s (%s), CV accuracy %.3f +- %.3f'
          % (best_set, best_clf, describe_params(best['gs'].best_params_), best['mean'], best['std']))
    plot_parameter_selection(table[(best_set, 'KNN')]['gs'], table[(best_set, 'SVM')]['gs'], best_set)
    X_best, _ = stack_blocks(feats, best['blocks'])
    plot_learning_curve(best['gs'].best_estimator_, X_best[idx_train], y_train, cv,
                        '%s + %s' % (best_set, best_clf))

    # ---------------- which classes each feature type separates ----------------
    analysis = cv_class_analysis(feats, idx_train, y_train, classes, table, best_clf)
    print_class_analysis(analysis, classes, best_clf)
    plot_class_analysis(analysis, classes, best_clf)
    show_figures(saved)

    # ---------------- test set ----------------
    # For comparison, the best configuration of every classifier type (by CV) is
    # trained on the full training set and scored once on the test set.
    print('\nTest set (%d images)' % len(idx_test))
    test_results = []
    for clf_name, _, _ in CLASSIFIERS:
        key = max((k for k in table if k != 'n_train' and k[1] == clf_name), key=lambda k: table[k]['mean'])
        X, _ = stack_blocks(feats, table[key]['blocks'])
        model = with_probabilities(table[key]['gs'].best_estimator_).fit(X[idx_train], y_train)
        m = evaluate(model, X[idx_test], y_test, classes)
        name = '%s (%s)' % (clf_name, key[0])
        test_results.append((name, m))
        print('  %-26s accuracy %.3f | macro F1 %.3f | top-2 %.3f | top-3 %.3f'
              % (name, m['accuracy'], m['macro_f1'], m['topk'][1], m['topk'][2]))
        if key == best_key:
            selected, selected_model = m, model
    print('\nSelected model on the test set:')
    print('  %-10s %9s %9s %9s' % ('class', 'precision', 'recall', 'F1'))
    for c, p, r, f in zip(classes, selected['precision'], selected['recall'], selected['f1']):
        print('  %-10s %9.3f %9.3f %9.3f' % (c, p, r, f))
    plot_test_results(selected, classes, 'Selected model: %s features + %s' % (best_set, best_clf))
    plot_topk(test_results, len(classes))
    plot_misclassified(images, y_test, rank_classes(selected_model, X_best[idx_test]), idx_test)
    show_figures(saved)

    # ---------------- final model ----------------
    final = with_probabilities(best['gs'].best_estimator_)
    if FINAL_TRAIN_ON_ALL:
        final.fit(X_best, labels)          # more data for the deployed model
    else:
        final = selected_model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({'model': final, 'blocks': best['blocks'], 'params': FEATURE_PARAMS,
                     'classes': classes, 'description': '%s + %s (%s)'
                     % (best_set, best_clf, describe_params(best['gs'].best_params_))}, f)
    print('\nSaved model to %s (trained on %d images)'
          % (MODEL_FILE, len(labels) if FINAL_TRAIN_ON_ALL else len(idx_train)))

    # check the deployed function on one image of each class
    print('assign2_sceneclassifier examples:')
    for c in classes:
        p = paths[idx_test[np.where(y_test == c)[0][0]]]
        print('  %-50s -> %s' % (p, assign2_sceneclassifier(p)[:3]))

    print('Figures saved to %s (total time %.0f s)' % (FIGURE_DIR, time.time() - t0))
    plt.ioff()
    plt.show()  # keep all figure windows open until they are closed


if __name__ == '__main__':
    main()
