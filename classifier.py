import numpy as np
import cv2
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os
from lesson_functions import *


def prepare_data(cspace='BGR', spatial_size=(32, 32),
                 hist_bins=32, hist_range=(0, 256),
                 orient=9, pix_per_cell=8, cell_per_block=2):

    images_path = glob.glob('../images/**/*.png', recursive=True)
    cars, notcars = get_cars_notcars(images_path)

    car_features = extract_features(cars, cspace, spatial_size,
                                    hist_bins, hist_range,
                                    orient, pix_per_cell, cell_per_block)
    notcar_features = extract_features(notcars, cspace, spatial_size,
                                       hist_bins, hist_range,
                                       orient, pix_per_cell, cell_per_block)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    y_labeled = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y_labeled, test_size=0.2, shuffle=True)

    return X_train, y_train, X_test, y_test


def measure_accuracy(clf, X_test, test_labels):
    acc = clf.score(X_test, test_labels)

    print("accuracy: ", acc)


def svm_classifier(X, y):
    clf = SVC(kernel='linear', C=50)
    clf.fit(X, y)

    return clf, "linear_svm"


if __name__ == "__main__":
    color_space = 'BGR'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    spatial_size = (32, 32)  # Spatial binning dimensions
    hist_bins = 32  # Number of histogram bins
    hist_range = (0, 256)
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block


    X_train, y_train, X_test, y_test = \
        prepare_data(cspace, spatial_size,
                     hist_bins, hist_range,
                     orient, pix_per_cell, cell_per_block)

    clf, classifier_name = svm_classifier(X_train, y_train)
    measure_accuracy(clf, X_test, y_test)

    # parameters as a dictionary
    parameters = {'color_space': color_space,
                  'spatial_size': spatial_size,
                  'hist_bins': hist_bins,
                  'hist_range': hist_range,
                  'orient': orient,
                  'pix_per_cell': pix_per_cell,
                  'cell_per_block': cell_per_block}

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
