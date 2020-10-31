import time
import efros


def main():
    img_names = ["T1.gif", "T2.gif", "T3.gif", "T4.gif", "T5.gif"]
    window_size = [5, 9, 11， 15， 17]
    for img in img_names:
        for i in window_size:
            print img
            start = time.time()
            ef = efros.Efors(i)
            ef.grow_Image(str(img), 200, 200, img.split('.')[0] + "_WS" + str(i) + ".gif")
            end = time.time()
            print "Texture {:s} \t Windows Size {:d} \t Time {:.2f} sec".format(img, i, (end-start))
            print "Finished in " + str(end - start) + " seconds"



if __name__ == "__main__":
    main()

