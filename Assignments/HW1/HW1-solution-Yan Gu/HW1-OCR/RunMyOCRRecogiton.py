import os
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
from collections import Counter
import test
import enhance_1_test
import enhance_5_test
import enhance_15_test
from pprint import pprint


def run_Recogiton(test_filename, pkl_filename, display_plot=False, enhance_num=None):
    file_path = './H1-16images/'
    # display_plot = True

    if enhance_num == None:
        Ypred, Coordinate, D_matrix = test.predict(test_filename, file_path, display_plot)
        accuracy = test.score(pkl_filename, Ypred, Coordinate)
    elif enhance_num == 1:
        Ypred, Coordinate, D_matrix = enhance_1_test.predict(test_filename, file_path, display_plot)
        accuracy = enhance_1_test.score(pkl_filename, Ypred, Coordinate)
    elif enhance_num == 5:
        Ypred, Coordinate, D_matrix = enhance_5_test.predict(test_filename, file_path, display_plot)
        accuracy = enhance_5_test.score(pkl_filename, Ypred, Coordinate)
    elif enhance_num == 15:
        Ypred, Coordinate, D_matrix = enhance_15_test.predict(test_filename, file_path, display_plot)
        accuracy = enhance_15_test.score(pkl_filename, Ypred, Coordinate)

    pprint(D_matrix)
    np.savetxt("result.txt", D_matrix)

    return accuracy


# for test
if __name__ == "__main__":
    file_path = './H1-16images/'
    display_plot = True
    test_1 = file_path + 'test1.bmp'
    test_2 = file_path + 'test2.bmp'
    test_result_1 = 'test1_gt.pkl'
    test_result_2 = 'test2_gt.pkl'

    accuracy_1 = run_Recogiton(test_1, test_result_1, display_plot)
    accuracy_2 = run_Recogiton(test_2, test_result_2, display_plot)

    total_accuracy = (accuracy_1 + accuracy_2) / 2.0

    print accuracy_1
    print accuracy_2