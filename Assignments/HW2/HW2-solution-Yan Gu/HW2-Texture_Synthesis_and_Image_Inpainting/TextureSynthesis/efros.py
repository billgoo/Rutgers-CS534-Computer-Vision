import numpy as np
import math
from skimage import io, morphology, exposure
from random import randint
import matplotlib.pyplot as plt
import time


class Efors:

    def __init__(self, window_size):
        self.ErrThreshold = 0.1
        self.MaxErrThreshold = 0.3
        self.Sigma = window_size / 6.4

        self.input_path="TextureSynthesis/images/"
        self.output_path="TextureSynthesis/results/"

        self.window_size = window_size
        self.window_area = window_size * window_size
        self.half_window_size = (window_size - 1) / 2


    def grow_Image(self, sample_img_name, new_image_row, new_image_col, output_name):
        start_time = time.time()
        
        sample = io.imread(str(self.input_path + sample_img_name))

        sample_img = sample / 255.00

        # plt.imshow(sample, cmap="gray")
        # plt.show()

        img_row, img_col = np.shape(sample_img)

        # idea of sliding windows and seed is inspired from the Internet
        img_window = self.sliding_Window(sample_img)
        print img_window.shape

        number_pixel = new_image_row * new_image_col
        
        synthetic_image = np.zeros((new_image_row, new_image_col))

        # Seed
        seed_size = 3
        random_row = randint(0, img_row - seed_size)
        random_col = randint(0, img_col - seed_size)
        seed = sample_img[random_row: random_row+seed_size, random_col: random_col+seed_size]
        synthetic_image[
            int(math.floor(new_image_row/2)): int(math.floor(new_image_row/2))+seed_size,
            int(math.floor(new_image_col/2)): int(math.floor(new_image_col/2))+seed_size
        ] = seed


        number_filled = seed_size * seed_size
        # unfilled pixels are black
        filled_list = np.zeros((new_image_row, new_image_col))
        filled_list[
            int(math.floor(new_image_row/2)): int(math.floor(new_image_row/2)) + seed_size,
            int(math.floor(new_image_col/2)): int(math.floor(new_image_col/2)) + seed_size
        ] = np.ones((seed_size, seed_size))
        
        padded_synthetic_image = np.lib.pad(synthetic_image, self.half_window_size, 'constant', constant_values=0)
        padded_filled_list = np.lib.pad(filled_list, self.half_window_size, 'constant', constant_values=0)
        
        gauss_mask = self.gauss2D(self.window_size, sigma=self.Sigma)
        print gauss_mask.shape

        while number_filled < number_pixel:
            progress = 0
            pixel_list = self.get_Unfilled_Neighbors(filled_list)
            
            for i in pixel_list:
                pixel_row = i[0]
                pixel_col = i[1]

                template = padded_synthetic_image[pixel_row : pixel_row+self.half_window_size+self.half_window_size+1,
                                                pixel_col : pixel_col+self.half_window_size+self.half_window_size+1]
                
                best_matches = self.find_Best_Matches(
                                        template,
                                        img_window,
                                        padded_filled_list[
                                                        pixel_row : pixel_row+self.half_window_size+self.half_window_size+1,
                                                        pixel_col : pixel_col+self.half_window_size+self.half_window_size+1
                                        ],
                                        np.reshape(gauss_mask, self.window_area)
                                    )
                
                best_match = randint(0, len(best_matches)-1)
                
                if best_matches[best_match][0] <= self.MaxErrThreshold:
                    padded_synthetic_image[self.half_window_size+pixel_row][self.half_window_size+pixel_col] = best_matches[best_match][1]
                    synthetic_image[pixel_row][pixel_col] = best_matches[best_match][1]
                    padded_filled_list[self.half_window_size+pixel_row][self.half_window_size+pixel_col] = 1
                    filled_list[pixel_row][pixel_col] = 1

                    number_filled += 1
                    if number_filled % math.ceil(number_pixel / 100) == 0:
                        print "Pixels filled {:d}/{:d} Time = {:3.2f} sec".format(number_filled, number_pixel, time.time()-start_time)
                    progress = 1
            if progress == 0:
                self.MaxErrThreshold *= 1.1
                print "new threshold = " + str(self.MaxErrThreshold)

        io.imsave(self.output_path + output_name, synthetic_image)
        # io.imshow(synthetic_image)
        # plt.show()


    def gauss2D(self, window_size, sigma):
        shape = (window_size, window_size)
        print sigma
        m, n = [(ss - 1.0) / 2.0 for ss in shape]
        y, x = np.ogrid[-m: m+1, -n: n+1]
        h = np.exp( -(x * x + y * y) / (2.0 * sigma * sigma) )
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h


    def sliding_Window(self, img):
        window_matrix = []
        for i in range(self.half_window_size, img.shape[0]-self.half_window_size-1):
            for j in range(self.half_window_size, img.shape[1]-self.half_window_size-1):
                window_matrix.append(
                    np.reshape(
                        img[i-self.half_window_size: i+self.half_window_size+1, 
                            j-self.half_window_size: j+self.half_window_size+1], 
                        (2 * self.half_window_size + 1) ** 2
                    )
                )
        return np.double(window_matrix)


    def find_Best_Matches(self, template, image_window, valid_mask, gauss_mask):
        template = np.reshape(template, self.window_area)
        valid_mask = np.reshape(valid_mask, self.window_area)
        total_weight = np.sum(np.multiply(gauss_mask, valid_mask))
        distance = (image_window - template) ** 2
        SSD = np.sum((distance * gauss_mask * valid_mask) / total_weight, axis=1)
        # print "Min err mat= "+str(min_error)
        center = ((2 * self.half_window_size + 1) ** 2) / 2
        return [[error, image_window[i][center]] for i, error in enumerate(SSD) if error <= min(SSD) * (1 + self.ErrThreshold)]


    def get_Unfilled_Neighbors(self, filled_image):
        candidate_pixel_row, candidate_pixel_col = np.nonzero(morphology.binary_dilation(filled_image) - filled_image)
        neighborhood = []
        for i in range(len(candidate_pixel_row)):
            pixel_row = candidate_pixel_row[i]
            pixel_col = candidate_pixel_col[i]
            neighborhood.append(
                np.sum(
                    filled_image[pixel_row-self.half_window_size: pixel_row+self.half_window_size+1,
                                pixel_col-self.half_window_size: pixel_col+self.half_window_size+1]
                )
            )
        # print candidate_pixel_row.shape
        order = np.argsort(-np.array(neighborhood, dtype=int))
        # print order
        result_list = []

        for x, i in enumerate(order):
                pixel_row = candidate_pixel_row[i]
                pixel_col = candidate_pixel_col[i]
                result_list.append([pixel_row, pixel_col])
        
        return result_list
