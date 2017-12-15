import numpy as np
from math import gamma
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
import glob
from lesson_functions import extract_features, get_cars_notcars
from sklearn.grid_search import GridSearchCV
import time


def measure_accuracy(clf, X_test, test_labels):
    acc = clf.score(X_test, test_labels)

    return acc


def svm_classifier(X, y):
    clf = LinearSVC()

    t = time.time()
    clf.fit(X, y)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train Linear SVC...')

    return clf, "linear_svm"


def svm_classifier_optimized(X, y):
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    svr = SVC()
    clf = GridSearchCV(svr, parameters)

    t = time.time()
    clf.fit(X, y)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train Optimized SVC...')
    print(clf.best_params_)

    return clf, "optimized_svm"


if __name__ == "__main__":
    images_path = glob.glob('../images/**/*.png', recursive=True)
    cars, notcars = get_cars_notcars(images_path)

    # sample_size = 500
    # cars = cars[0:sample_size]
    # notcars = notcars[0:sample_size]

    cspace = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = "ALL"  # Can be 0, 1, 2, or ALL
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 16  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    y_start_stop = [None, None]  # Min and max in y to search in slide_window()

    car_features = extract_features(cars, color_space=cspace,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=cspace,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    clf, classifier_name = svm_classifier(X_train, y_train)

    # Check the prediction time for a single sample
    print('Test Accuracy of SVC = ', measure_accuracy(clf, X_test, y_test))

    # parameters as a dictionary
    parameters = {'color_space': cspace,
                  'orient': orient,
                  'pix_per_cell': pix_per_cell,
                  'cell_per_block': cell_per_block,
                  'hog_channel': hog_channel,
                  'spatial_size': spatial_size,
                  'hist_bins': hist_bins,
                  'spatial_feat': spatial_feat,
                  'hist_feat': hist_feat,
                  'hog_feat': hog_feat}

    # Save the data for easy access
    pickle_file = 'classifier.p'
    if not os.path.isfile(pickle_file):
        print('Saving data to pickle file...')
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                {
                    'X_scaler': X_scaler,
                    'parameters': parameters,
                    'clf': clf,
                    'name': classifier_name
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
