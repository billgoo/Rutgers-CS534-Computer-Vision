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
from enhance_1_train import get_Train_Features


def get_Test_Features(filename, display_plot=False):
    
    th = 200
    result = {}
    Features = []
    coordinate = []
    img = io.imread(filename)

    # print img.shape

    # latest version of numpy remove np.double
    img_binary = (img < th).astype(np.double)
    result['img_binary'] = img_binary
        
    img_label = label(img_binary, background=0, connectivity=1) # neighbors default = 8 use connectivity=1 neighbors=4
    # output number of labeled sub image
    # print np.amax(img_label)

    # show plots
    if display_plot:
        io.imshow(img)
        plt.title('Original Image')
        io.show()

        hist = exposure.histogram(img)
        plt.bar(hist[1], hist[0])
        plt.title('Histogram')
        plt.show()
            
        io.imshow(img_binary)
        plt.title('Binary Image')
        io.show()

        io.imshow(img_label)
        plt.title('Labeled Image')
        io.show()

    regions = regionprops(img_label)
    # find the threshold used to remove the small noise
    thre_noise = {'height':[10.0, 80.0], 'width':[12.0, 85.0]}
    io.imshow(img_binary)
    ax = plt.gca()
    for props in regions:
        # coordinate of the pixels
        minr, minc, maxr, maxc = props.bbox

        # use if to remove too small or too large region
        if (maxr - minr < thre_noise['height'][0] or maxc - minc < thre_noise['width'][0] 
        or maxr - minr > thre_noise['height'][1] or maxc - minc > thre_noise['width'][1]):
            continue

        if display_plot:
            ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, 
                        edgecolor='red', linewidth=1))
        #print minc, minr, maxc, maxr, maxr-minr, maxc-minc
        roi = img_binary[minr:maxr, minc:maxc]
        m = moments(roi)
        cr = m[0, 1] / m[0, 0]
        cc = m[1, 0] / m[0, 0]
        mu = moments_central(roi, cr, cc)
        nu = moments_normalized(mu)
        hu = moments_hu(nu)

        Features.append(hu)
        coordinate.append([minr, minc, maxr, maxc])

    if display_plot:
        plt.title('Bounding Boxes')
        io.show()

    result['Features'] = Features
    result['Coordinate'] = coordinate

    return result


def normalized(features, mean, std):
    # normalization
    feature_array = np.array(features)
    norm_features = []
    for i in range(feature_array.shape[0]):
        feature_array[i] -= mean
        feature_array[i] /= std
        norm_features.append(feature_array[i])

    return norm_features


def predict(test_filename, file_path, display_plot):
    train_features, Label, mean, std = get_Train_Features(file_path, display_plot)
    dataset = get_Test_Features(test_filename, display_plot)

    # normalization for test data features
    test_features = normalized(dataset['Features'], mean, std)

    D = cdist(test_features, train_features)
    if display_plot:
        io.imshow(D)
        plt.title('Distance Matrix')
        io.show()

    D_index = np.argsort(D, axis=1)
    Ypred = [Label[i[0]] for i in D_index]
    if display_plot:
        draw_TestCC_Image(dataset['img_binary'], Ypred, dataset['Coordinate'])

    return Ypred, dataset['Coordinate'], D


def draw_TestCC_Image(img_binary, Ypred, Coordinate):

    io.imshow(img_binary)
    ax = plt.gca()    
    for i in range(len(Ypred)):
        y = Ypred[i]
        # coordinate of the pixels
        c = Coordinate[i]
        minr, minc, maxr, maxc = c[0], c[1], c[2], c[3]

        ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, 
                    edgecolor='red', linewidth=1))
        plt.text(maxc, minr, y, bbox=dict(facecolor='yellow', alpha=0.5))
        # inner for end
    plt.title('Bounding Boxes for test image')
    io.show()


def score(verify_filename, Ypred, Coordinate):
    # load the data in the result
    file_pkl = open(verify_filename, 'rb')
    data_pkl = pickle.load(file_pkl)
    file_pkl.close()
    location = data_pkl['locations']
    Ytrue = data_pkl['classes']
    
    num_correct = 0.0
    num_total = len(Ytrue)
    for i in range(num_total):
        for j in range(len(Coordinate)):
            cenc = location[i][0]
            cenr = location[i][1]
            minr, minc = Coordinate[j][0], Coordinate[j][1]
            maxr, maxc = Coordinate[j][2], Coordinate[j][3]
            if cenr < minr or cenr > maxr or cenc < minc or cenc > maxc:
                continue

            if Ytrue[i] == Ypred[j]:
                num_correct += 1
                # print location[i], Coordinate[j]
                break
    accuracy_rate = num_correct / num_total
    # print num_total, len(Ypred)

    return accuracy_rate



if __name__ == "__main__":
    file_path = './H1-16images/'
    display_plot = True
    test_1 = file_path + 'test1.bmp'
    test_2 = file_path + 'test2.bmp'
    test_result_1 = 'test1_gt.pkl'
    test_result_2 = 'test2_gt.pkl'

    Ypred_1, Coordinate_1, D_1 = predict(test_1, file_path, display_plot)
    Ypred_2, Coordinate_2, D_2 = predict(test_2, file_path, display_plot)

    accuracy_1 = score(test_result_1, Ypred_1, Coordinate_1)
    accuracy_2 = score(test_result_2, Ypred_2, Coordinate_2)

    total_accuracy = (accuracy_1 + accuracy_2) / 2.0

    print accuracy_1
    print accuracy_2
    print total_accuracy
