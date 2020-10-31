from __future__ import division # uncomment this if using Python 2.7
import numpy as np
from scipy import spatial
from scipy.ndimage import gaussian_filter, sobel
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage import data, color, img_as_float, io
from skimage.morphology import erosion, disk
import time
from math import *


def get_patches(img, points, size_w):
    """
    - img: (n, m) input RGB image
    - points: (n, m) position of corners
    - w: integer patch size
    """
    patches=[]
    img3=np.lib.pad(img, ((size_w//2, size_w//2),(size_w//2,size_w//2)), 'edge')
    for i in points:
        patches.append(np.array(img3[i[0]:i[0]+2*size_w//2, i[1]:i[1]+2*size_w//2].flatten()))
        
    return patches


def spectral_matching(sim, patches1, patches2, maskinv, corner_pos1,corner_pos2):
    """define a similarity measure between Euclidean and SSD and correlation"""

    patches2 = [x*maskinv for x in patches2]
    
    patches1=patches1*maskinv
    matchall=spatial.distance.cdist(patches1, patches2, sim)
    
    match=[]
    for i in range(matchall.shape[0]):
        match.append((corner_pos1, corner_pos2[np.argmin(matchall[i])]))

    return match


def compute_Pivot(border_I, confidence, newImg):
    for idx, i in enumerate(border_I):
        patchblur=get_patches(imgblur, [i], size_w)[0].reshape((size_w,size_w))
        patchmask=get_patches(masknew, [i], size_w)[0].reshape((size_w,size_w))
        dx = sobel(patchblur, axis=0)
        dy = -sobel(patchblur, axis=1)
        nx = sobel(patchmask, axis=0)
        ny = sobel(patchmask, axis=1) 

        mod = np.sqrt(dx ** 2 + dy ** 2)
        modI=(np.unravel_index(mod.argmax(), mod.shape))
        
        v1=np.array([dx[modI[0],modI[1]],dy[modI[0],modI[1]]])
        v2=np.array([ny[half_size_w,half_size_w],nx[half_size_w,half_size_w]]).T
        
        D = abs(np.dot(v1,v2/np.linalg.norm(v2)))
        D /= (sqrt(dy[modI[0], modI[1]] ** 2 + dx[modI[0], modI[1]] ** 2) *
        sqrt(nx[half_size_w,half_size_w] ** 2 + ny[half_size_w,half_size_w] ** 2))
        
        border_C[idx]=confidence[i[0],i[1]]*D
        
    indx=np.argmax(border_C)
    maxI = border_I[indx]
    return maxI


if __name__ == "__main__":
    img_names = "test_im3.bmp"
    input_path="images/"
    output_path="results/"
    output_name = ["result1.bmp"]
    mask_name = ["mask1.bmp"]
    patchSize = [11]

    for imi in range(1):
        for size_w in patchSize:
            print mask_name[imi]
            RGB_img1 = data.imread(str(input_path+img_names))
            img1 = img_as_float(color.rgb2gray(RGB_img1))

            RGB_mask = data.imread(str(input_path+mask_name[imi]))
            mask = img_as_float(color.rgb2gray(RGB_mask))

            '''
            plt.figure(figsize=(12,6))
            plt.subplot(121)
            io.imshow(RGB_img1, cmap=cm.gist_gray)
            plt.show()
            plt.title('Shrub L')
            plt.subplot(122)
            io.imshow(RGB_mask, cmap=cm.gist_gray)
            plt.show()
            plt.title('Shrub R');
            '''
            half_size_w = size_w // 2

            masknew=mask.copy()
            newImg = img1.copy()
            imgblur = gaussian_filter(newImg, sigma=1, order=0)
            max_x_shape, max_y_shape = newImg.shape
            # io.imsave(output_path + output_name, newImg)

            start_time = time.time()
  
            patchesIndex = [
                    (ix,iy) for ix, row in enumerate(newImg) for iy, i in enumerate(row) if (ix>size_w and iy>size_w and ix<newImg.shape[0]-size_w and iy<newImg.shape[1]-size_w and ix%half_size_w==0 and iy%half_size_w==0) and (1 not in get_patches(mask, [(ix,iy)], size_w)[0])
                ]
            patches = get_patches(newImg, patchesIndex, size_w)

            confidence = 1. - masknew.copy()

            border = masknew - erosion(masknew, disk(1))
            border_I =  [(ix,iy) for ix, row in enumerate(border) for iy, i in enumerate(row) if i==1]  
            border_C=[0] * len(border_I)

            
            for idx, i in enumerate(border_I):
                confidence[i[0],i[1]]=sum(get_patches(confidence, [i], size_w)[0])/size_w**2
                
            while (masknew==1).sum()>0 :
                border = masknew - erosion(masknew, disk(1))
                border_I = [(ix,iy) for ix, row in enumerate(border) for iy, i in enumerate(row) if i==1 ]  
                border_C = [0] * len(border_I)
                
                pivotI = compute_Pivot(border_I, confidence, newImg)
                
                print("Pixels remain to solve: ")
                print((masknew == 1).sum())

                pivotPatch = get_patches(newImg, [pivotI], size_w)

                maskpatch = -(get_patches(masknew, [pivotI], size_w)[0]-1)
                match = spectral_matching('euclidean', pivotPatch, patches, maskpatch, pivotI, patchesIndex)

                pivotI_xmin=match[0][0][0]-half_size_w
                pivotI_xmax=match[0][0][0]+half_size_w
                pivotI_ymin=match[0][0][1]-half_size_w
                pivotI_ymax=match[0][0][1]+half_size_w
                
                match_xmin=match[0][1][0]-half_size_w
                match_xmax=match[0][1][0]+half_size_w
                match_ymin=match[0][1][1]-half_size_w
                match_ymax=match[0][1][1]+half_size_w

                if pivotI_xmin < 0:
                    match_xmin -= pivotI_xmin
                    pivotI_xmin = 0
                elif pivotI_ymin < 0:
                    match_ymin -= pivotI_ymin
                    pivotI_ymin = 0
                if pivotI_xmax > max_x_shape:
                    match_xmax = match_xmax - pivotI_xmax + max_x_shape
                    pivotI_xmax = max_x_shape
                elif pivotI_ymax > max_y_shape:
                    match_ymax = match_ymax - pivotI_ymax + max_y_shape
                    pivotI_ymax = max_y_shape

                newImg[pivotI_xmin:pivotI_xmax, pivotI_ymin:pivotI_ymax]=newImg[match_xmin:match_xmax, match_ymin:match_ymax]
                masknew[pivotI_xmin:pivotI_xmax, pivotI_ymin:pivotI_ymax]=0

                pivotI_xmin = match[0][0][0]-half_size_w
                pivotI_xmax = match[0][0][0]+half_size_w
                pivotI_ymin = match[0][0][1]-half_size_w
                pivotI_ymax = match[0][0][1]+half_size_w
                
                match_xmin = match[0][1][0]-half_size_w
                match_xmax = match[0][1][0]+half_size_w
                match_ymin = match[0][1][1]-half_size_w
                match_ymax = match[0][1][1]+half_size_w
                
                if pivotI_xmin < 0:
                    match_xmin -= pivotI_xmin
                    pivotI_xmin = 0
                elif pivotI_ymin < 0:
                    match_ymin -= pivotI_ymin
                    pivotI_ymin = 0
                if pivotI_xmax > max_x_shape:
                    match_xmax = match_xmax - pivotI_xmax + max_x_shape
                    pivotI_xmax = max_x_shape
                elif pivotI_ymax > max_y_shape:
                    match_ymax = match_ymax - pivotI_ymax + max_y_shape
                    pivotI_ymax = max_y_shape
                
                confidence[pivotI_xmin:pivotI_xmax, pivotI_ymin:pivotI_ymax]=confidence[match_xmin:match_xmax, match_ymin:match_ymax]

            print("--- %s seconds ---" % (time.time() - start_time))
            '''
            plt.figure(figsize=(12,6))
            io.imshow(newImg, cmap='gray')
            plt.show()
            plt.title('Superimage')
            '''
            io.imsave(output_path + output_name[imi], newImg)