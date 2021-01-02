import timeit
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image, ImageEnhance
from scipy import ndimage
import scipy.ndimage
import cv2
import cv2 as cv
import math
import scipy.misc


start = timeit.timeit()

numberOfImage = 31

while numberOfImage <= 44:
    infile = r'C:\Users\Mario\Desktop\Testowe\4\DSC_00%s.jpg' % (numberOfImage)
    outfile = r"C:\Users\Mario\Desktop\Testowe\4\Edited\DSC_%d.jpg" % numberOfImage

    img1 = cv2.imread(infile, 1)

    # img1 = cv2.imread(r"C:\Users\Mario\Desktop\test3.jpg", 1)
    # height, width = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    # img1[:,:,1] = 0 #green to zero

    plt.imshow(img1)
    # plt.show()

    img1 = cv2.fastNlMeansDenoising(img1)
    plt.imshow(img1)
    # plt.show()

    hsv = cv.cvtColor(img1, cv.COLOR_BGR2HSV)

    green = np.uint8([[[255, 0, 255]]])
    hsv_green = cv.cvtColor(green, cv.COLOR_BGR2HSV)
    print(hsv_green)

    lower_blue = np.array([0, 0, 200])
    upper_blue = np.array([10, 255, 255])

    mask = cv.inRange(hsv, lower_blue, upper_blue)

    res = cv.bitwise_and(img1, img1, mask=mask)

    print("Mean of V")
    hmean = np.mean(hsv[:, :, 2])
    print(hmean)
    print("Var of V")
    hvar = np.var(hsv[:, :, 2])
    print(hvar)
    print("Std of V")
    hstd = np.std(hsv[:, :, 2])
    print(hstd)

    plt.imshow(hsv[:, :, 2])
    # plt.show()

    # To save
    toSave = hsv[:, :, 2] > (hmean + hstd)
    # plt.imshow(toSave)
    # plt.savefig(outfile)

    im = Image.fromarray(toSave)
    im.save(outfile)

    # plt.show()

    # cv2.imwrite(r'numberOfImage.jpg', toSave)

    # cv2.imwrite(outfile, toSave)
    # print(numberOfImage)

    numberOfImage = numberOfImage + 1

print("gotowe")

# End of the loop


plt.imshow(hsv[:, :, 2] > (hmean + 2 * hstd))
plt.show()
plt.imshow(hsv[:, :, 2])
plt.show()
plt.imshow(mask)
plt.show()
plt.imshow(res)
plt.show()

##########################
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret, gray = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
plt.imshow(gray)
plt.show()

blurred = cv2.GaussianBlur(gray, (15, 15), 0)
plt.imshow(blurred)
plt.show()

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(gray, (15, 15), 0)
plt.imshow(blur)
plt.show()
ret3, th3 = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(th3)
plt.show()

ret, th1 = cv2.threshold(blurred, 7, 255, cv2.THRESH_BINARY)
plt.imshow(th1)
plt.show()

thresh = cv2.threshold(img1, 100, 255, cv2.THRESH_BINARY)[1]
plt.imshow(thresh)
plt.show()

thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)
plt.imshow(thresh)
plt.show()

# ret3, th3 = cv.threshold(blur, 240, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# plt.subplot(1,3,1), plt.hist(img1)
# plt.subplot(1,3,2), plt.hist(blurred)
# plt.subplot(1,3,3), plt.hist(thresh)

# plt.show()
##################################


# edges = cv2.Canny(img1, 200, 255)
# plt.imshow(img1)
# plt.show()


# laplacian = cv2.Laplacian(img1,cv2.CV_64F)
# plt.imshow(laplacian)
# plt.show()

# sobely = cv2.Sobel(img1,cv2.CV_64F,0,1,ksize=5)
# plt.imshow(sobely)
# plt.show()


# dst = cv2.fastNlMeansDenoisingMulti(img1, 2, 5, None, 4, 7, 35)

# dst = cv2.fastNlMeansDenoisingColored(img1,None,10,10,7,21)
# plt.imshow(dst)
# plt.show()

# kernel = np.ones((5, 5), np.uint8)
# opening = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)
# plt.imshow(opening)
# plt.show()

# blur = cv.GaussianBlur(opening, (5, 5), 0)
# ret3, th3 = cv.threshold(blur, 240, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# plt.subplot(1,3,1), plt.imshow(img1)
# plt.subplot(1,3,2), plt.imshow(opening)
# plt.subplot(1,3,3), plt.imshow(th3)

# plt.show()


sobelx8u = cv2.Sobel(img1, cv2.CV_8U, 0, 1, ksize=5)
plt.imshow(sobelx8u)
plt.show()

median = cv2.medianBlur(sobelx8u, 5)
plt.imshow(median)
plt.show()

blur = cv2.bilateralFilter(median, 15, 75, 75)
plt.imshow(blur)
plt.show()

blur = cv2.bilateralFilter(median, 15, 200, 255)
plt.imshow(blur)
plt.show()

# result =
# output = result.binarize(90).invert()
# plt.imshow(output)
# plt.show()


result = ndimage.gaussian_laplace(blur, sigma=5)
plt.imshow(result)
plt.gray()
plt.show()

blur = cv2.bilateralFilter(median, 50, 30, 150)
plt.imshow(blur)
plt.show()

kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(sobelx8u, cv2.MORPH_OPEN, kernel)
print("dede")
plt.imshow(opening)
plt.show()

print("teraz")

##########################
gray = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
plt.imshow(blurred)
plt.show()

thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]
plt.imshow(thresh)
plt.show()

thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)
plt.imshow(thresh)
plt.show()

# ret3, th3 = cv.threshold(blur, 240, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# plt.subplot(1,3,1), plt.hist(img1)
# plt.subplot(1,3,2), plt.hist(blurred)
# plt.subplot(1,3,3), plt.hist(thresh)

# plt.show()
##################################


result = ndimage.gaussian_laplace(opening, sigma=5)
plt.imshow(result)
plt.gray()
plt.show()

hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

lower_red = np.array([30, 150, 50])
upper_red = np.array([255, 255, 180])

mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(img1, img1, mask=mask)

# plt.imshow(img1)
# plt.show()
# plt.imshow(mask)
# plt.show()
# plt.imshow(res)
# plt.show()

# img = cv2.imread(r"C:\Users\Mario\Desktop\test1.png", 0)
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

plt.imshow(thresh)
# plt.show()

edges = blur

edges = cv2.Canny(edges, 20, 255)
plt.imshow(img1)
# plt.show()

edges = cv2.Canny(edges, 245, 255)
plt.imshow(img1)
# plt.show()

edges = cv2.Canny(edges, 250, 255)
plt.imshow(img1)
# plt.show()

end = timeit.timeit()
print(end - start)

plt.imshow(img1)

# plt.show()
