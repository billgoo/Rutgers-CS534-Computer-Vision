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


def get_Train_Features(file_path, display_plot=False):
    # file_path = './H1-16images/'
    # find all files with names
    filenames = []
    # validate in Unix or Windows platform
    for root_path, child_dirs, files in os.walk(file_path):
        for f in files:
            if f[0:4] != 'test':
                filenames.append(f)

    th = 200
    Features = []
    Label = []
    for fname in filenames:
        img = io.imread(file_path + fname)

        # print img.shape

        # latest version of numpy remove np.double
        img_binary = (img < th).astype(np.double)
        
        img_label = label(img_binary, background=0)
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

            # change to other feature to accerate
            Features.append(hu)
            Label.append(fname[0])
            # inner for end
        if display_plot:
            plt.title('Bounding Boxes')
            io.show()

    # normalization
    feature_array = np.array(Features)
    expect = np.mean(feature_array, axis=0, dtype=np.double)
    s_deviation = np.std(feature_array, axis=0, dtype=np.double)
    norm_Features = []
    for i in range(feature_array.shape[0]):
        feature_array[i] -= expect
        feature_array[i] /= s_deviation
        norm_Features.append(feature_array[i])

    if display_plot:
        draw_CC_Image(file_path, norm_Features, Label, expect, s_deviation)

    return norm_Features, Label, expect, s_deviation


def draw_CC_Image(file_path, Features, Label, mean, std):
    # file_path = './H1-16images/'
    # find all files with names
    filenames = []
    # validate in Unix or Windows platform
    for root_path, child_dirs, files in os.walk(file_path):
        for f in files:
            if f[0:4] != 'test':
                filenames.append(f)

    th = 200
    num_correct = 0.0
    num_total = len(Label)
    
    for fname in filenames:
        img = io.imread(file_path + fname)
        img_binary = (img < th).astype(np.double)
        
        img_label = label(img_binary, background=0)

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

            # change to other feature to accerate
            norm_hu = hu - mean
            norm_hu /= std
            D = cdist([norm_hu], Features)
            # print D

            D_index = np.argsort(D, axis=1)
            # get the 2nd index of each row
            Ypred = Label[D_index[0][1]]
            if fname[0] == Ypred:
                num_correct += 1
            plt.text(maxc, minr, Ypred, bbox=dict(facecolor='yellow', alpha=0.5))
            # inner for end
        t = 'Bounding Boxes: ' + fname[0]
        plt.title(t)
        io.show()
    accuracy_rate = num_correct / num_total
    # print accuracy_rate


# use to test in this file
def train_predict(file_path, display_plot):
    Features, Label, mean, std = get_Train_Features(file_path, display_plot)
    D = cdist(Features, Features)
    # print D
    if display_plot:
        io.imshow(D)
        plt.title('Distance Matrix')
        io.show()

    D_index = np.argsort(D, axis=1)
    # get the 2nd index of each row
    Ypred = [Label[i[1]] for i in D_index]

    num_correct = 0.0
    num_total = len(Ypred)
    for i in range(num_total):
        if Label[i] == Ypred[i]:
            num_correct += 1
    accuracy_rate = num_correct / num_total
    print accuracy_rate

    confM = confusion_matrix(Label, Ypred)
    if display_plot:
        io.imshow(confM)
        plt.title('Confusion Matrix')
        io.show()

    return Ypred, confM, accuracy_rate



if __name__ == "__main__":
    file_path = './H1-16images/'
    # for test
    # display_plot = False
    display_plot = True
    Ypred, confM, accuracy = train_predict(file_path, display_plot)

