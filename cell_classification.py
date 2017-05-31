# start step-by step
print ("hello")
from skimage import io
from skimage.external import tifffile # io for tif-files
import numpy as np

import scipy.ndimage as scp # for 3D features and filters

from sklearn.externals import joblib # to save and load the classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score

import timeit
from threading import active_count
import multiprocessing as mp 

print("importing libraries: done")

# some constants
# defines how many gaussian rings you take
sigmas=[2, 4]
is3D = False
num_procs = 8 

print("# of available threads:", active_count())
print("# of available procs:", mp.cpu_count())
print("# of procs set:", num_procs)

mp.cpu_count()

# extended to 3D feature generation
def generate_features3D(image, sigma):
    # generate range of sigmas
    sigmas = range(sigma[0], sigma[1] + 1)

    f_values = image.flatten()
    f_sobel = scp.sobel(image).flatten()

    f_gauss = np.zeros([len(image.flatten()), len(sigmas)])
    f_dog = np.zeros([len(image.flatten()), len(sigmas) - 1])

    idx = 0
    for s in range(sigma[0], sigma[1] + 1):
        # consider only Re part for gabor filter
        f_gauss[:, idx] = scp.gaussian_filter(image, s).flatten()
        if (idx != 0):
            f_dog[:, idx - 1] = f_gauss[:, idx] - f_gauss[:, idx - 1]
        idx += 1

    f_max = scp.maximum_filter(image, sigma[0]).flatten()
    f_median = scp.median_filter(image, sigma[0]).flatten() # run median only with the minimal sigma
    f_laplacian = scp.laplace(image).flatten()

    # full set of features
    f_set = np.vstack([f_values, f_max,
                       f_median, f_sobel,
                       f_gauss.T, f_dog.T,
                       f_laplacian]).T
    return f_set


# read the raw data and the binary classification (test data)

if is3D:
    raw = io.imread('data/raw-3D.tif')
    cells = io.imread('data/binary-3D.tif')
else:
    raw = io.imread('data/raw-49.tif')
    cells = io.imread('data/binary-49.tif')
bg = np.invert(cells)
print("reading images: done")

# set 0 to be background and 1 to the cells
cells[cells == 255] = 1
np.unique(cells)

y = cells.flatten()
X = generate_features3D(raw, sigmas)
print("generating features: done")
print(X.shape[1], "features will be used")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X.shape, X_train.shape, X_test.shape
print("generating data: done")

skf = StratifiedKFold(n_splits=5)
precision_scores = list()
recall_scores = list()
aucs = list()


start_time = timeit.default_timer()

k_fold_idx = 0
for train_ix, test_ix in skf.split(X, y): # for each of K folds
    # define training and test sets
    X_train, X_test = X[train_ix,:], X[test_ix,:]
    y_train, y_test = y[train_ix], y[test_ix]

    # Train classifier
    clf = RandomForestClassifier(n_jobs=num_procs)
    clf.fit(X_train, y_train)

    # Predict test set labels
    yhat = clf.predict(X_test)
    yprob = clf.predict_proba(X_test)

    # Calculate metrics
    aucs.append(roc_auc_score(y_test, yprob[:,1]))
    precision_scores.append(precision_score(y_test, yhat))
    recall_scores.append(recall_score(y_test, yhat))

    print ("K-fold iteration", k_fold_idx, ": done")
    k_fold_idx += 1

elapsed = timeit.default_timer() - start_time
print("training classifier: done in", elapsed, "sec")

# save the classifier to the file
# sys.setrecursionlimit(10000) # but you might need this one
joblib.dump(clf, "data/clf.pkl", compress=9)
print("saving classifier: done")

# restore the classifier from the file
# clf = joblib.load('data/clf.pkl')
# print("loading classifier: done")

# prepare run on the real data
raw_real = io.imread('data/raw-85.tif')

# calculate the features manually
X_real = generate_features3D(raw_real, sigmas);
print("generating features: done")

raw_real_predicted = clf.predict(X_real)
raw_real_predicted_proba = clf.predict_proba(X_real)
print("generating features: done")

result = np.reshape(raw_real_predicted, raw_real.shape)
# to save as 32-bit tif
result_proba = np.ndarray(shape=(raw_real.shape), dtype=np.float32)
result_proba[()] = np.reshape(raw_real_predicted_proba[:, 1], raw_real.shape)


tifffile.imsave('data/raw.tif', raw_real)
tifffile.imsave('data/result.tif', result)
tifffile.imsave('data/proba.tif', result_proba)

print("saving results: done")

print("script: done")
