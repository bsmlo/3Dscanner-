import matplotlib.image as img
import scipy.misc as mic

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from PIL import Image
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#i = Image.open(r"C:\Users\Mario\Desktop\test.png")
#iar = np.asarray(i)

#print(iar)

img1 = mpimg.imread(r"C:\Users\Mario\Desktop\test.png")

gray = rgb2gray(img1)

#plt.imshow(gray, cmap = plt.get_cmap('gray'))


grayimg = np.asarray(gray)



#print(grayimg)

print(grayimg)

plt.matshow(grayimg)

print(np.nonzero(grayimg))

print(np.average(grayimg[np.nonzero(grayimg)], weights=np.nonzero(grayimg)))


plt.show()



