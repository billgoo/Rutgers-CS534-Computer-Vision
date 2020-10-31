import time
import efros


def main():
    img_names = ["test_im1.bmp", "test_im2.bmp"]
    #img_names = ["test_im3_1.bmp", "test_im3_2.bmp", "test_im3_3.bmp"]
    window_size = [5, 9, 11, 15, 17]
    #window_size = [11]
    for img in img_names:
        for i in window_size:
            print img
            start = time.time()
            ef = efros.Efors(i)
            ef.grow_Image(str(img), img.split('.')[0] + "_WS" + str(i) + ".bmp")
            end = time.time()
            print "Texture {:s} \t Windows Size {:d} \t Time {:.2f} sec".format(img, i, (end-start))
            print "Finished in " + str(end - start) + " seconds"



if __name__ == "__main__":
    main()

