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
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import pyqtgraph.exporters

import pyqtgraph.examples

#pyqtgraph.examples.run()

numberOfImage = 31

a = np.array([])
b = np.array([])
c = np.array([])
color = np.array([])

while numberOfImage <= 44:


    infile = r"C:\Users\Mario\Desktop\Testowe\4\Edited\DSC_%d.jpg" %numberOfImage

    img1 = cv2.imread(infile, 0)

    img_array = np.asarray(img1)

    #print(img_array.shape)

    if numberOfImage == 31:
        a, b = np.nonzero(img_array)
        c = np.full(a.shape, (numberOfImage-31)*50, dtype=int)
        #print(a.shape)
        #print(c.shape)

    else:
        aa, bb = np.nonzero(img_array)
        cc = np.full(aa.shape, (numberOfImage-31)*50, dtype=int)
        a = np.append(a, aa)
        b = np.append(b, bb)
        c = np.append(c, cc)
        #a = np.dstack([a, img_array])

    numberOfImage = numberOfImage + 1

maximb = max(b)
color1 = [b[i]/maximb for i in b]
color = np.asarray(color1)
#print(color[131])

all_points = np.stack((a, b, c), axis=1)
#print(all_points.shape)



#fig = plt.figure()
#ax = plt.axes(projection='3d')
#x,y,z = a.nonzero()
#ax.scatter(x, y, z*10, c=-x+y-z)
#plt.show()


import pyqtgraph as pg
pg.mkQApp()
#pg.plot(x, y, pen=None, symbol='o')
#pg.show()



## make a widget for displaying 3D objects
import pyqtgraph.opengl as gl
view = gl.GLViewWidget()
view.show()


#print(x.shape)

## create three grids, add each to the view
#xgrid = gl.GLGridItem()
#ygrid = gl.GLGridItem()
#zgrid = gl.GLGridItem()

#plot = gl.GLScatterPlotItem
#view.addItem(xgrid)
#view.addItem(ygrid)
#view.addItem(zgrid)

#pos = np.random.randint(-10,10,size=(5,3))
#pos[:,2] = np.abs(pos[:,2])

sp2 = gl.GLScatterPlotItem(pos=all_points)
view.addItem(sp2)




## rotate x and y grids to face the correct direction
#xgrid.rotate(90, 0, 1, 0)
#ygrid.rotate(90, 1, 0, 0)

## scale each grid differently
#xgrid.scale(0.2, 0.1, 0.1)
#ygrid.scale(0.2, 0.1, 0.1)
#zgrid.scale(0.1, 0.2, 0.1)


## Start Qt event loop.
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(pg.QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()