# This class can edit image using few filters or procedures
import numpy as np
import cv2
from skimage import img_as_float
from skimage import color, morphology, exposure
from kneed import KneeLocator
from PIL import Image
from skimage.morphology import white_tophat, skeletonize


import concurrent.futures


class ImageFilters:

    def __init__(self):
        self.images_to_edit = []
        self.edited_images = []
        self.compare_images = []  # images before motphological thinnig
        self.image_matrix = [[], [], []]
        self.list_of_functions = []
        self.layer_number = 0
        self.resolution = 0  # 1.82608   #42cm/number of frames to change in image container

    # Geometrical undistorion
    def filter_undistort(self, image):
        mtx = ([1.43007971e+03, 0.00000000e+00, 6.23310543e+02],
               [0.00000000e+00, 1.41523006e+03, 3.20646989e+02],
               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00])

        dist = ([-4.17384627e-01, 1.50022972e-01, -3.18096773e-04, 9.81290115e-04,
                 1.45909991e-01])

        #    w = src.shape[1]
        #    h = src.shape[0]
        newcameramtx = ([1.24925537e+03, 0.00000000e+00, 6.21223086e+02],
                        [0.00000000e+00, 1.23910193e+03, 3.17573223e+02],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00])

        # undistortion
        dst = cv2.undistort(np.float32(image), np.float32(mtx), np.float32(dist), None, np.float32(newcameramtx))
        dst = dst.astype(np.uint8)

        return dst

    # Binarization, get matrix from one layer picture
    def binearization(self, image):
        img_binarization = np.asarray(image)

        # check shape of image colur or grayscale
        if len(img_binarization.shape) >= 3:
            try:
                # print("***converting to grayscale***")
                gray = cv2.cvtColor(np.float32(img_binarization), cv2.COLOR_BGR2GRAY)
                bin = np.nonzero(gray)

                # add matrix to image container
                # convertion to cm y/34.909090 x/58.272727
                self.add_matrix([bin[1] / 60.884, bin[0] / 31.597])

            except ValueError:
                print(ValueError)
                print("Can't get the matrix")
        else:
            try:
                # print("***Collecting data..***")
                bin = np.nonzero(img_binarization)
                # add matrix to image container
                # convertion to cm y/34.909090 x/58.272727
                self.add_matrix([bin[1] / 60.884, bin[0] / 31.597])
            except ValueError:
                print(ValueError)
                print("Can't get the matrix")

    # Binarization, get matrix from one layer picture
    def binearize(self, image):
        img_binarization = np.asarray(image)
        shape = img_binarization.shape

        if len(shape) >= 3:
            try:
                img_binarization = cv2.cvtColor(np.float32(img_binarization), cv2.COLOR_BGR2GRAY)

            except ValueError:
                print(ValueError)
                print("Can't transform the image")

        try:
            bin = np.nonzero(img_binarization)
            img_mtx = [bin[1] / 60.884, (shape[0] - bin[0]) / 31.597]

            return img_mtx

            # self.add_matrix([img_mtx[0], img_mtx[1]])

        except ValueError:
            print(ValueError)
            print("Can't get the matrix")

    # Perspective correction
    def perspective_correction(self, image):
        img_perspective = np.asarray(image)

        if len(img_perspective.shape) == 3:
            rows, cols, ch = img_perspective.shape
        else:
            # print('gray')
            rows, cols = img_perspective.shape

        # Transformation matrix from getPerspectiveTransform
        M = [[4.61386962e-01, -5.65776917e-01, 3.24411613e+02],
             [7.19903651e-04, 3.07245847e-01, 8.75850391e+01],
             [1.25418754e-06, -9.41057451e-04, 1.00000000e+00]]

        undistorted = cv2.warpPerspective(np.float32(img_perspective), np.float32(M),
                                          (np.float32(cols), np.float32(rows)))

        undistorted = (undistorted).astype(np.uint8)

        return undistorted

    #  Filter with fast denoising
    def filter_denoising(self, image):
        img_filter_b = np.asarray(image)

        if len(img_filter_b.shape) >= 3:
            try:
                # print("***Colour denoising***")
                img_filter_b = cv2.fastNlMeansDenoisingColored(img_filter_b, None, 20, 20, 7,
                                                               15)  # Denoising colour image
                return img_filter_b
            except ValueError:
                print(ValueError)
                print("Can't denoise this image-try undo")
                input("press any key to back to processing menu...")
                return img_filter_b
        else:
            try:
                print("***Grayscale denoising***")
                img_filter_b = cv2.cv2.fastNlMeansDenoising(image)  # Denoising grayscale image
                return img_filter_b
            except ValueError:
                print(ValueError)
                print("Can't denoise this image-try undo")
                input("press any key to back to processing menu...")
                return img_filter_b

    #  Filtering specific channel of HSV between values
    def rgb_range_filter(self, image):
        img_rgb_filter = np.asarray(image)
        #arr = np.asarray(image)

        if len(img_rgb_filter.shape) >= 3:
            try:
                # print("***Colour filtering***")
                get_rgb = self.findrgbrange(img_rgb_filter)

                rgb_range = [(get_rgb[0], 255), (get_rgb[1], 255), (get_rgb[2], 255)]

                # Data in BGR
                red_range = np.logical_and(rgb_range[0][0] <= img_rgb_filter[:, :, 0], img_rgb_filter[:, :, 0] <= rgb_range[0][1])
                green_range = np.logical_and(rgb_range[1][0] <= img_rgb_filter[:, :, 1], img_rgb_filter[:, :, 1] <= rgb_range[1][1])
                blue_range = np.logical_and(rgb_range[2][0] <= img_rgb_filter[:, :, 2], img_rgb_filter[:, :, 2] <= rgb_range[2][1])

                '''
                    # Data in BGR
                    red_range = np.logical_and(rgb_range[0][0] <= arr[:, :, 0], arr[:, :, 0] <= rgb_range[0][1])
                    green_range = np.logical_and(rgb_range[1][0] <= arr[:, :, 1], arr[:, :, 1] <= rgb_range[1][1])
                    blue_range = np.logical_and(rgb_range[2][0] <= arr[:, :, 2], arr[:, :, 2] <= rgb_range[2][1])'''

                    #redgreen = np.logical_or(red_range, green_range)
                    #redgreenblue = np.logical_or(redgreen, blue_range)

                redgreen = np.logical_or(red_range, green_range)

                redgreenblue = np.logical_or(redgreen, blue_range)

                img_rgb_filter[np.logical_not(redgreenblue), 0] = 0
                img_rgb_filter[np.logical_not(redgreenblue), 1] = 0
                img_rgb_filter[np.logical_not(redgreenblue), 2] = 0



                #arr[np.logical_not(redgreenblue), 0] = 0
                #arr[np.logical_not(redgreenblue), 1] = 0
                #arr[np.logical_not(redgreenblue), 2] = 0

                #import matplotlib.pyplot as plt
                #for i in range(0, 3):
                #    plt.imshow(Image.fromarray(arr[i]))
                #    plt.show()

                #ready = []

                #ready[np.logical_and(redgreen, True)] = 255

                #arr[np.logical_not(redgreen), 0] = 0
                #arr[np.logical_not(redgreen), 1] = 0
                #arr[np.logical_not(redgreen), 2] = 0

                #print(redgreen)

                #arr[np.logical_not(redgreen), 0] = 0
                #arr[np.logical_not(redgreen), 1] = 0
                #arr[np.logical_not(redgreen), 2] = 0

                #import matplotlib.pyplot as plt
                #plt.imshow(Image.fromarray(redgreen, mode='L'))
                #plt.show()

                out_image = Image.fromarray(redgreen, mode='L')

                return out_image

            except ValueError:
                print(ValueError)
                print("Can't filter this image")

                return image

        else:
            try:
                # print("***Grayscale filtering***")
                get_rgb = self.findrgbrange(img_rgb_filter)
                gray_range = np.logical_and(get_rgb[0] <= img_rgb_filter[:], img_rgb_filter[:] <= 255)

                img_rgb_filter[np.logical_not(gray_range)] = 0

                out_image = Image.fromarray(img_rgb_filter)

                return out_image

            except ValueError:
                print(ValueError)
                print("Can't filter this image")

                return image

                #  Morphological operations

    #to gray
    def togray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return gray


    #  Thinning binarized after morph and perspective correction
    def thinning(self, image):
        # print("***Thinning***")
        thin_image = img_as_float(image)

        kernel_dil = np.ones((3, 3), np.uint8)

        image_binary = morphology.closing(thin_image)
        image_binary = cv2.dilate(np.float32(image_binary), kernel_dil, iterations=3)
        out_skel = skeletonize(image_binary, method='lee')

        '''
        thin_image = img_as_float(image)

        kernel_dil = np.ones((3, 3), np.uint8)

        image_binary = morphology.opening(thin_image)
        image_binary = cv2.dilate(np.float32(image_binary), kernel_dil, iterations=3)
        out_skel = skeletonize(image_binary, method='lee')'''

        return out_skel

    #  Morphological operations can be used without denoising - but part of the data can be lost
    def morphology_filter(self, image):
        morph_img = np.asarray(image)

        kernel_dil = np.ones((3, 3), np.uint8)

        if len(morph_img.shape) >= 3:
            try:
                # print("***converting to grayscale***")
                morph_for_ski = img_as_float(color.rgb2gray(morph_img))

                # binearisation
                image_binary = morph_for_ski > 0.001
                morph_for_ski[np.logical_not(image_binary)] = 0

                # binearization - Black tophat
                wth = white_tophat(morph_img)

                ch_one = wth[:, :, 0] > 5
                ch_two = wth[:, :, 1] > 5
                ch_three = wth[:, :, 2] > 5

                morph_for_ski[ch_one] = 0
                morph_for_ski[ch_two] = 0
                morph_for_ski[ch_three] = 0

                return (morph_for_ski * 255).astype(np.uint8)

            # image_binary = morphology.opening(morph_for_ski)
            # image_binary = cv2.dilate(np.float32(image_binary), kernel_dil, iterations=3)
            # out_skel = skeletonize(image_binary, method='lee')

            except ValueError:
                print(ValueError)
                print("Can't convert this image to grayscale")
                return image

        # good work with denoising, gamma, sv from hsv and knee filtering
        # this is much more precisly but, takes much longer time
        else:
            try:
                print("***Collecting data..***")
                morph_for_ski = img_as_float(morph_img)
                # binearisation
                image_binary = morph_for_ski > 0.001
                morph_for_ski[np.logical_not(image_binary)] = 0

                return (morph_for_ski * 255).astype(np.uint8)

                # image_binary = morphology.opening(morph_for_ski)
                # image_binary = cv2.dilate(np.float32(image_binary), kernel_dil, iterations=3)
                # out_skel = skeletonize(image_binary, method='lee')

                # return out_skel

            except ValueError:
                print(ValueError)
                print("Can't convert this image to grayscale")
                return image

    # gausian blur
    def gausianblur(self, image):
        image = cv2.bilateralFilter(image, 10, 50, 50)
        return image

    # Append matrix with new layer
    def add_matrix(self, image_matrix):
        # XY_arrays = [[],[]]
        lenght = len(image_matrix[0])
        # self.layer_number * 0.8, image_matrix[0]

        for i in range(0, lenght):
            self.image_matrix[0].append(self.layer_number * self.resolution)
            self.image_matrix[1].append(image_matrix[0][i])
            self.image_matrix[2].append(image_matrix[1][i])
        # self.image_matrix[0].append(XY_arrays)

        self.layer_number += 1

    # Append matrix with a new layer
    def make_matrix(self, image_matrix, layer_number):
        lenght = len(image_matrix[0])

        for i in range(0, lenght):
            self.image_matrix[0].append(layer_number * self.resolution)
            self.image_matrix[1].append(image_matrix[0][i])
            self.image_matrix[2].append(image_matrix[1][i])

    #  check final number of images
    def checkimages(self, lenofthelist):
        if lenofthelist == len(self.edited_images):
            return True
        else:
            return False

    #  save edited image to the list
    def imageappend(self, image):
        self.edited_images.append(image)

    # return the list of edited images
    def returnimages(self):
        return self.edited_images, self.image_matrix

    # finding range for rgb range filter
    def findrgbrange(self, image):
        RGB = []
        # Computing the knees for rgb
        if len(image.shape) >= 3:
            try:
                # print("***Colour RGB range detection***")
                for i in range(0, 3):
                    cdf3 = exposure.cumulative_distribution(image[:, :, i])

                    mask = tuple([cdf3[0] > 0.95])

                    #print(cdf3[1])

                    filtred = tuple( [cdf3[0][mask],  cdf3[1][mask] ] )

                    #print(filtred[1])



                    #import matplotlib.pyplot as plt
                    #plt.figure(1)

                    dff = np.diff(filtred[0])
                    ex = (np.array(filtred[1])[:-1] + np.array(filtred[1])[1:]) / 2
                    #print(len(ex))
                    #print(len(dff))
                    #plt.imshow(image[:, :, i])
                    #plt.show()
                    #plt.clf()
                    #plt.plot(ex, dff)

                    #

                    kneedle = KneeLocator(ex, dff, S=30, curve='convex', direction='increasing')
                    #kneedle.

                    #kneedle.plot_knee()
                    #plt.show()
                    RGB.append(int(kneedle.knee))


                return RGB

            except ValueError:
                print(ValueError)
                print("Can't get the range")
                input("press any key to back to processing menu...")
                return False

        else:
            try:
                # print("***Grayscale  range detection***")
                cdf3 = exposure.cumulative_distribution(image)
                kneedle = KneeLocator(cdf3[1], cdf3[0], curve='convex', direction='increasing')
                RGB.append(int(kneedle.knee))
                return RGB

            except ValueError:
                print(ValueError)
                print("Can't get the range")
                input("press any key to back to processing menu...")
                return False


# sequence from imagelist
class Sequence(ImageFilters):
    def call_sequence(self, listofimage, listoffunctions, resolution):
        image_number = 0
        print(len(listofimage))
        self.resolution = resolution

        # self.list_of_functions = listoffunctions

        for image in listofimage:
            print(image_number)
            for function in listoffunctions:
                if function == 'barell':
                    image = self.filter_undistort(image)
                elif function == 'perspective':
                    image = self.perspective_correction(image)
                elif function == 'denoising':
                    image = self.filter_denoising(image)
                elif function == 'rgbrange':
                    image = self.rgb_range_filter(image)
                elif function == 'morph':
                    image = self.morphology_filter(image)
                elif function == 'gausian':
                    image = self.gausianblur(image)
                elif function == 'binearization':
                    self.binearization(image)
                elif function == 'thinning':
                    image = self.thinning(image)
                elif function == 'range':
                    image = self.rgb_range_filter(image)
                else:
                    print('no ' + str(function) + ' function')

            image_number = image_number + 1
            self.imageappend(image)

        if self.checkimages(len(listofimage)):
            image_number = 0
            print('done')
            self.returnimages()
        else:
            print('Error! Not every image converted...')
            self.returnimages()

    # ['barell', 'rgbrange', 'morph', 'perspective', 'thinning', 'binearization']
    # Process the all the images from the list in threads
    def CallSequenceThreads(self, listofimage, resolution):
        image_number = 0
        print(str(len(listofimage)) + "callseq")
        self.resolution = resolution

        # Complete processing for one image
        def sequenceOneImage(image, img_number):
            print(img_number)
            image = image
            image = self.filter_undistort(image)  # Barell undst
            image = self.rgb_range_filter(image)  # RGB range
            image = self.morphology_filter(image)  # Morph
            image = self.perspective_correction(image)  # Perspective
            image = self.thinning(image)  # Thinning
            layer_mtx = self.binearize(image)  # Binearization
            self.make_matrix(layer_mtx, img_number)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(0, len(listofimage) - 1):
                executor.submit(sequenceOneImage, listofimage[i], (len(listofimage) - 1) - i)

    # load_paths and return one image from specific patch
    # def loadOneImagge(self, patch):
    #    img = cv2.imread(patch)
    #    return img
