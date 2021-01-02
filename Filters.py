import sys
import os
import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imagecontainer as ic
import choices as clt
from PIL import Image
from skimage import img_as_float
from skimage import io, color, morphology, exposure
from skimage.morphology import skeletonize, thin, erosion, dilation, opening, \
    closing, white_tophat, black_tophat, skeletonize, convex_hull_image
import ImageFilters as IF
from ImageFilters import Sequence
from skimage.restoration import denoise_nl_means, estimate_sigma
from mpl_toolkits import mplot3d

import ImageFilters

file_path = ""  # String with image path
img = cv2.imread("")  # Image variable

#  Elements for load_paths, undo and redo options
img_list = []  # Image collection
actual_loaded = 0  # Index of actual processing image

# Variables for hsv processing menu
channel_to_save = False  # Used in hsv_processing_menu
selected_channel = cv2.imread("")  # Actual loaded channel

# Image container
# imageContainer = ic.ImageContainer()

# Class operations
import Operations as Operation

Function = Operation.Operations()

import ImageFilters as fltrs

filters = fltrs.ImageFilters()

# Edit handler


# list of functions used by user
choices = clt.Choices()

new_path = r'C:\Users\Mario\Desktop\image_20.jpg'


# Main Menu
def main_menu():
    global file_path
    global test

    # Load images using operations class
    Function.load_images()
    # Function.image_show()

    # Image container test
    # Load all images from selected path
    #    if imageContainer.number_of_elements == 0:
    #        imageContainer.load_paths(r'E:\Data\Dokumenty\Studia\Praca\MGR\Serie\23-11-1', r'.jpg')  ### loading off
    #        imageContainer.current_image_list.append(imageContainer.image_list[0])  ### load_paths image to show
    #    else:
    #        print("Images already loaded to the list...")
    #        print(imageContainer.number_of_elements)
    # imageContainer.save()

    while True:
        print("************Menu************")

        # No path info and manu content set
        if False:  # file_path == "":

            print("***No file path defined!***")

            # Menu content set
            content = """1 - Chose image path  
x - Exit:"""

        else:
            content = """1 - Chose image path
2 - Show image     
3 - Image processing
4 - Load NPY
5 - Chose image from loaded list
x - Exit:"""

        # Main options
        # Red user's chose
        choice = input(content)

        if choice == "1":
            chose_path()

        elif choice == "2":  # and file_path != "": # dorobic sprawdzanie jak obrazy nie sa zaladowane
            # show_img(0)
            Function.image_show()
        elif choice == "3":  # and img.size != 0: # dorobic sprawdzanie jak obrazy nie sa zaladowane
            processing_menu()
        elif choice == "4":
            load_npy()
        elif choice == "5":
            # chose single image to edit
            Function.image_to_edit()
        elif choice == "x" or choice == "X":
            menu_exit()
        else:
            print("Wrong input")
            main_menu()


# Processing Menu
def processing_menu():
    while True:
        print("************Processing Menu************")

        content = """1 - Show original image 
2 - Filter fast denoising   
3 - Change to HSV
4 - Normalise
5 - Histogram
6 - Undistortion
7 - Blur
8 - Skimage nl fast denoise
9 - Gamma corectiono test
T - Thinning
B - Between values filter
L - Binearisation
P - Perspective correction
M - Morphological operation
O - Gausian otsus filtering
G - 2 gray
w - Save picture
R - Run sequence
V - Visualisation
s - Show
u - Undo
r - Redo
x - Back to Main Menu:"""

# 34 34

        # Main options
        # Red user's chose
        choice = input(content)

        if choice == "1":
            Function.image_show()
            show_img(0)  # Show original image
        elif choice == "2" and file_path != "":
            filter_denoising()
        elif choice == "3" and img.size != 0:
            rgb_to_hsv()
        elif choice == "4" and img.size != 0:
            normalise_img()
        elif choice == "5" and img.size != 0:
            show_histogram()
        elif choice == "6":  # and img.size != 0:
            Function.add_to_stack(filters.filter_undistort(Function.actual_image()))
            # filter_undistort()
        elif choice == "7" and img.size != 0:
            gausianblur()
        elif choice == "8" and img.size != 0:
            skinldenoise()
        elif choice == "9" and img.size != 0:
            gammacorection()
        elif choice == "T" or choice == "t":
            Function.add_to_stack(filters.thinning(Function.actual_image()))
        elif choice == "B" or choice == "b":
            Function.add_to_stack(filters.rgb_range_filter(Function.actual_image()))
            # rgb_range_filter()
        elif choice == "M" or choice == "m":
            morphology_filter()
        elif choice == "G" or choice == "g":
            Function.add_to_stack(filters.togray(Function.actual_image()))
            # togray()
        elif choice == "O" or choice == "o":
            gausianotsus()
        elif choice == "P" or choice == "p":
            Function.add_to_stack(filters.perspective_correction(Function.actual_image()))
            # perspective_correction()
        # elif choice == "L" or choice == "l":
        # binearization()
        elif choice == "W" or choice == "w":
            save_picture()
        elif choice == "R":
            Function.dosequenceThreads()

            # load_paths images from the list

            # imageContainer.dosequence(['barell', 'rgbrange', 'morph', 'perspective', 'thinning', 'binearization'])

            # trzeba najpierd odfitrować morph, później korekcja perspektywy i dopieto thinning bo powstają rozlania

            # try to save image matrix
            Function.save_matrix()

        elif choice == "V" or choice == "v":
            Operation.visualisation()

            # visualisation()

        elif choice == "u" or choice == "U":  # Undo last filter
            Function.undo_edit()
            # undo_edit()
        elif choice == "r":  # Redo last filter
            Function.redo_edit()
            # redo_edit()
        elif choice == "s" or choice == "S":  # Show actual loaded image
            Function.image_show()
            # show_img(actual_loaded)
        elif choice == "x" or choice == "X":
            main_menu()
        else:
            print("Wrong input")
            processing_menu()


def hsv_processing_menu():
    while True:
        global selected_channel
        global channel_to_save  # Gets true if you chose specific channel

        # Aditional information when specyfic chanel is selected
        if channel_to_save:
            aditional = "\nF - Standard deviation channel filtering" \
                        "\nB - Filter between specific values"
        else:
            aditional = ""

        print("************HSV Processing************")

        content = """SH - Show H channel 
SS - Show S channel   
SV - Show V channel%s
S - Save selected channel
x - Cancel:""" % aditional

        # Main options
        # Read user's chose
        choice = input(content)

        if choice == "SH" or choice == "sh":
            selected_channel = img_list[actual_loaded][:, :, 0]  # Load H from HSV
            channel_to_save = True
            show_one_img(selected_channel)
        elif choice == "SS" or choice == "ss":
            selected_channel = img_list[actual_loaded][:, :, 1]  # Load S from HSV
            channel_to_save = True
            show_one_img(selected_channel)
        elif choice == "SV" or choice == "sv":
            selected_channel = img_list[actual_loaded][:, :, 2]  # Load V from HSV
            channel_to_save = True
            show_one_img(selected_channel)
        elif choice == "F" or choice == "f":
            factor = input("Insert the factor:")
            sf_filter(selected_channel, factor)
            channel_to_save = True
        elif choice == "B" or choice == "b":
            min_value = input("Insert the value (min) ")
            max_value = input("Insert the value (max) ")
            hsv_range_filter(selected_channel, min_value, max_value)
            channel_to_save = True
        elif choice == "s" or choice == "S":
            if channel_to_save:
                add_to_stack(selected_channel)
                channel_to_save = False  # Default value
                selected_channel = cv2.imread("")  # Delete image from variable
                processing_menu()
            else:
                print("No specyfic channel selected-full HSV already saved")
        elif choice == "x" or choice == "X":
            channel_to_save = False  # Default value
            selected_channel = cv2.imread("")  # Delete image from variable
            processing_menu()
        else:
            print("Wrong input")
            hsv_processing_menu()


# Perspective correction
def perspective_correction():
    img_perspective = load_a_picture()
    # print(len(img_perspective.shape))

    # tu jest problem z rozpakowaniem gdy mamy obraz rgb lub grayscale trzeba wykrywac czy jest i rozpakować do 2 lub
    # 3 zmiennych
    if len(img_perspective.shape) == 3:
        rows, cols, ch = img_perspective.shape
    else:
        rows, cols = img_perspective.shape

    # Transformation matrix from getPerspectiveTransform
    M = [[4.61386962e-01, -5.65776917e-01, 3.24411613e+02],
         [7.19903651e-04, 3.07245847e-01, 8.75850391e+01],
         [1.25418754e-06, -9.41057451e-04, 1.00000000e+00]]

    undistorted = cv2.warpPerspective(np.float32(img_perspective), np.float32(M), (np.float32(cols), np.float32(rows)))

    undistorted = undistorted.astype(np.uint8)  ## protect data cliping

    add_to_stack(undistorted)
    show_img(actual_loaded)
    processing_menu()


# Close the program
def menu_exit():
    # Proof question EXIT
    print("Do you really want to exit? Y/N")
    final_chose = input()
    print(final_chose)
    if final_chose == "Y" or final_chose == "y":
        sys.exit()
    else:
        main_menu()


#  Load NPY as image matrix
def load_npy():
    Function.load_matrix()


# Chose path for image
def chose_path():
    global file_path  # String with image path
    global img  # Image variable

    #  Elements for load_paths, undo and redo options
    global img_list  # Image collection
    global actual_loaded  # Index of actual processing image

    # Reset variables
    file_path = ""
    img = cv2.imread("")
    img_list = []
    actual_loaded = 0

    new_path = ''

    print("""**Chose the path**
    Press x to cancel
    """)

    while True:
        new_path = input('Please insert the path:')
        if new_path == "x" or new_path == "X":
            break
        elif os.path.isdir(new_path):
            Function.images_patch = new_path
            Function.load_images()
            break
        else:
            print("Path does not exist!")


'''                
        if os.path.isfile(new_path):
            if new_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                
                file_path = new_path
                load_image()
            else:
                file_path = ""
                print("Wrong extension! Use: png, jpg or bmp.")
                chose_path()
        else:
            file_path = ""
            print("The file path does not exist!")
'''


# Load image from the path
def load_image():
    global img
    img = cv2.imread(file_path)
    img_list.append(img)


# Binarisation
# def binearization():
#    img_binarization = load_a_picture()
# gray = cv2.cvtColor(img_binarization, cv2.COLOR_BGR2GRAY)
# bin = np.nonzero(gray)
# plt.plot(bin[1], bin[0], '.')

# plt.plot(red_range = np.logical_and(bin[1] <= 537, arr[:, :, 0] <= rgb_range[0][1]))
# print(bin[1])
# print(bin[0])

# plt.show()
# processing_menu()
"""
    if len(img_binarization.shape) >= 3:
        try:
            print("***converting to grayscale***")
            gray = cv2.cvtColor(np.float32(img_binarization), cv2.COLOR_BGR2GRAY)
            bin = np.nonzero(gray)
            imageContainer.add_matrix([bin[1] / 60.884, bin[0] / 31.597])  # add matrix to image container
            plt.plot(bin[1] / 60.884, bin[0] / 31.597, '.')  # convertion to cm y/34.909090 x/58.272727
            ##test
            print(imageContainer.image_matrix)
            print(imageContainer.image_matrix[0][1])
            plt.plot(imageContainer.image_matrix[0][1], imageContainer.image_matrix[0][2], '.')
            plt.show()
            processing_menu()
        except ValueError:
            print(ValueError)
            print("Can't convert this image to grayscale")
            input("press any key to back to processing menu...")
            processing_menu()
    else:
        try:
            print("***Collecting data..***")
            bin = np.nonzero(img_binarization)
            imageContainer.add_matrix([bin[1] / 60.884, bin[0] / 31.597])  # add matrix to image container
            plt.plot(bin[1] / 60.884, bin[0] / 31.597, '.')  # convertion to cm y/34.909090 x/58.272727
            plt.axis('square')
            plt.show()
            processing_menu()
        except ValueError:
            print(ValueError)
            print("Can't plot this image...")
            input("press any key to back to processing menu...")
            processing_menu()
"""


# gamma corection
def gammacorection():
    image = load_a_picture()

    # ima = exposure.adjust_gamma(image, 2)
    ima = exposure.adjust_sigmoid(image, 0.8)

    add_to_stack(ima)
    show_img(actual_loaded)
    processing_menu()


# finding range for rgb range filter
def findrgbrange(image):
    from kneed import KneeLocator
    RGB = []
    print(image.shape)
    # Computing the knees for rgb
    if len(image.shape) >= 3:
        try:
            print("***Colour RGB range detection***")
            for i in range(0, 3):
                cdf3 = exposure.cumulative_distribution(image[:, :, i])

                kneedle = KneeLocator(cdf3[1], cdf3[0], curve='convex', direction='increasing')

                print(round(kneedle.knee, 3))
                RGB.append(int(kneedle.knee))
                print(round(kneedle.knee_y, 3))
            # plt.show()
            return RGB

        except ValueError:
            print(ValueError)
            print("Can't get the range")
            input("press any key to back to processing menu...")
            return False

    else:
        try:
            print("***Grayscale  range detection***")
            cdf3 = exposure.cumulative_distribution(image)
            # print(image.shape)
            # print(cdf3)
            kneedle = KneeLocator(cdf3[1], cdf3[0], curve='convex', direction='increasing')
            # kneedle.plot_knee()

            print(round(kneedle.knee, 3))
            RGB.append(int(kneedle.knee))
            print(round(kneedle.knee_y, 3))

            return RGB

        except ValueError:
            print(ValueError)
            print("Can't get the range")
            input("press any key to back to processing menu...")
            return False


# Show the image from img_list
def show_img(image_number):
    print(image_number)
    img_to_show = img_list[image_number]
    print(type(img_to_show))
    # img_to_show = cv2.cvtColor(img_list[int(image_number)], cv2.COLOR_BGR2RGB)
    plt.imshow(img_to_show)
    plt.show()


# Show single image
def show_one_img(image):
    # img_to_show = cv2.cvtColor(cv2.UMat(image), cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()


#  Processing menu
def image_processing():
    clear()


#  Clear the view
def clear():
    os.system('cls')


#  Load picture function from saved list
def load_a_picture():
    global actual_loaded
    loaded_image = img_list[actual_loaded]
    return loaded_image


#  Filter with fast denoising
def filter_denoising():
    img_filter_b = load_a_picture()

    if len(img_filter_b.shape) >= 3:
        try:
            import time
            start = time.time()

            print("***Colour denoising***")
            img_filter_b = img_list[actual_loaded]
            img_filter_b = cv2.fastNlMeansDenoisingColored(img_filter_b, None, 20, 20, 7, 15)  # Denoising colour image
            end = time.time()
            print(end - start)

            add_to_stack(img_filter_b)
            show_img(actual_loaded)
        except ValueError:
            print(ValueError)
            print("Can't denoise this image-try undo")
            input("press any key to back to processing menu...")
            processing_menu()
    else:
        try:
            print("***Grayscale denoising***")
            img_filter_b = cv2.cv2.fastNlMeansDenoising(img_filter_b)  # Denoising grayscale image
            add_to_stack(img_filter_b)
            show_img(actual_loaded)
        except ValueError:
            print(ValueError)
            print("Can't denoise this image-try undo")
            input("press any key to back to processing menu...")
            processing_menu()


# Geometrical undistorion
def filter_undistort():
    img_distortion = load_a_picture()
    # src = img_undistortion

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
    dst = cv2.undistort(np.float32(img_distortion), np.float32(mtx), np.float32(dist), None, np.float32(newcameramtx))

    dst = dst.astype(np.uint8)  ## protect data cliping

    add_to_stack(dst)
    show_img(actual_loaded)
    processing_menu()


# gausian blur
def gausianblur():
    blur = load_a_picture()
    blur = cv2.bilateralFilter(blur, 10, 50, 50)
    add_to_stack(blur.astype(np.uint8))
    show_img(actual_loaded)
    processing_menu()


# to gray
def togray():
    color = load_a_picture()
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    add_to_stack(gray)
    show_img(actual_loaded)
    processing_menu()


def skinldenoise():
    noisy = load_a_picture()

    patch_kw = dict(patch_size=5,  # 5x5 patches
                    patch_distance=6,  # 13x13 search area
                    multichannel=True)

    sigma_est = np.mean(estimate_sigma(noisy, multichannel=True))

    denoise2_fast = denoise_nl_means(noisy, h=0.6 * sigma_est, sigma=sigma_est,
                                     fast_mode=True, **patch_kw)
    add_to_stack(denoise2_fast)
    show_img(actual_loaded)
    processing_menu()


# gausian otsu filtering
def gausianotsus():
    src = load_a_picture()
    otsus_img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(otsus_img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    add_to_stack(th3)
    show_img(actual_loaded)
    processing_menu()


#  Morphological operations can be used without denoising - but part of the data can be lost
def morphology_filter():
    morph_img = load_a_picture()

    kernel = ([0, 1, 0],
              [0, 0, 0],
              [0, 0, 0])

    kernel = np.uint8(kernel)

    kernel_dil = np.ones((3, 3), np.uint8)
    kernel_ero = np.ones((7, 7), np.uint8)

    if len(morph_img.shape) >= 3:
        try:
            #### TEST
            print("***converting to grayscale***")

            morph_for_ski = ((color.rgb2gray(morph_img)) * 255).astype(np.uint8)

            wth = white_tophat(morph_img)

            show_one_img(wth)

            filter = np.logical_or(wth[:, :, 0] > 5, wth[:, :, 1] > 5)
            filter = np.logical_or(filter, wth[:, :, 2] > 5)

            morph_for_ski[filter] = 0

            add_to_stack(morph_for_ski)
            show_img(actual_loaded)
            processing_menu()

            #### TEST

            print("***converting to grayscale***")
            morph_for_ski = img_as_float(color.rgb2gray(morph_img))

            # binearisation
            image_binary = morph_for_ski > 0.001
            org = image_binary  # onlly to compare with edited
            morph_for_ski[np.logical_not(image_binary)] = 0
            show_one_img(morph_for_ski)

            # binearization - Black tophat
            wth = white_tophat(morph_img)
            # bth = black_tophat(morph_for_ski)#> 0.001

            ch_one = wth[:, :, 0] > 5
            ch_two = wth[:, :, 1] > 5
            ch_three = wth[:, :, 2] > 5

            morph_for_ski[ch_one] = 0
            morph_for_ski[ch_two] = 0
            morph_for_ski[ch_three] = 0

            letsee = 2 * org + 4 * morph_for_ski

            # show_one_img(morph_for_ski)
            # show_one_img(wth[:, :, 0])
            # show_one_img(wth[:, :, 1])
            # show_one_img(wth[:, :, 2])
            show_one_img(letsee)

            image_binary = morphology.opening(morph_for_ski)
            image_binary = cv2.dilate(np.float32(morph_for_ski), kernel_dil, iterations=3)
            out_skel = skeletonize(image_binary, method='lee')
            # out_thin = thin(image_binary)
            # show_one_img(image_binary + out_skel + out_thin)

            outlet = 5 * letsee + out_skel
            show_one_img(outlet)

            add_to_stack(out_skel)
            show_img(actual_loaded)
            processing_menu()

        except ValueError:
            print(ValueError)
            print("Can't convert this image to grayscale")
            input("press any key to back to processing menu...")
            processing_menu()

    # good work with denoising, gamma, sv from hsv and knee filtering
    # this is much more precisly but, takes much longer time
    else:
        try:
            print("***Collecting data..***")
            morph_for_ski = img_as_float(morph_img)
            # binearisation
            image_binary = morph_for_ski > 0.001
            org = image_binary  # onlly to compare with edited
            morph_for_ski[np.logical_not(image_binary)] = 0
            # show_one_img(morph_for_ski)

            # binearization - Black tophat
            # wth = white_tophat(morph_img)
            # show_one_img(wth)

            image_binary = morphology.opening(morph_for_ski)
            image_binary = cv2.dilate(np.float32(image_binary), kernel_dil, iterations=3)
            out_skel = skeletonize(image_binary, method='lee')
            add_to_stack(out_skel)
            show_img(actual_loaded)
            processing_menu()

        except ValueError:
            print(ValueError)
            print("Can't plot this image...")
            input("press any key to back to processing menu...")
            processing_menu()

    morph_kernel = ([0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0])  # np.ones((10, 10), np.uint8)


#  Adding to the stack of images
def add_to_stack(img_to_add):
    print(len(img_list))

    # Convert image to np ndarray class
    img_array_to_add = np.asarray(img_to_add)
    global actual_loaded
    #  Check if loaded image is the last one on the stack or is there oly original image
    if actual_loaded + 1 == len(img_list) or len(img_list) == 1:
        img_list.append(img_array_to_add)
        actual_loaded = actual_loaded + 1
        print(len(img_list))
    #  If not edited image will be add as next to loaded, all higher will be deleted
    else:
        img_list[actual_loaded + 1] = img_array_to_add
        actual_loaded = actual_loaded + 1
        len_of_img_stack = len(img_list)
        if actual_loaded + 1 < len_of_img_stack:
            del img_list[-(len_of_img_stack - (actual_loaded + 1)):]
        print(len(img_list))


#  Undo last image edit
def undo_edit():
    global actual_loaded
    if actual_loaded > 0:
        choices.undo_choice()  # remove last function
        actual_loaded = actual_loaded - 1
        show_img(actual_loaded)
    else:
        print("Can't undo.")
        processing_menu()


#  Redo last image edit
def redo_edit():
    global actual_loaded
    if 0 <= actual_loaded < len(img_list) - 1:
        choices.redo_choice()  # Readd last deleted function to the list
        actual_loaded = actual_loaded + 1
        show_img(actual_loaded)
    else:
        print("Can't redo.")
        processing_menu()


# save picture to orginal destination
def save_picture():
    img_to_save = load_a_picture()
    head, tail = os.path.split(new_path)
    file_path = head + '\edited_' + tail
    print(file_path)
    cv2.imwrite(file_path, img_to_save)


#  Load image as HSV
def rgb_to_hsv():
    img_to_hsv = np.asarray(load_a_picture())

    if len(img_to_hsv.shape) >= 3:
        try:
            img_to_hsv = img_list[actual_loaded]
            hsv = cv2.cvtColor(img_to_hsv, cv2.COLOR_BGR2HSV)
            add_to_stack(hsv)
            hsv_processing_menu()
        except ValueError:
            print(ValueError)
            print("Can't convert to hsv, try undo.")
            input("press any key to back to processing menu...")
            processing_menu()
    else:
        print("Can't convert rgb to hsv.")


#  Normalise an image
def normalise_img():
    img_to_normalise = load_a_picture()
    img_to_normalise_out = np.zeros((img_to_normalise.shape[0], img_to_normalise.shape[1]))
    img_to_normalise = cv2.normalize(img_to_normalise, img_to_normalise_out, 0, 255, cv2.NORM_MINMAX)
    add_to_stack(img_to_normalise)
    show_img(actual_loaded)
    processing_menu()


#  Filtering specific channel of HSV using standard deviation
def sf_filter(img_stat, factor):
    mean = np.mean(img_stat)
    std_dev = np.std(img_stat)
    print("mean:%s  Std:%s" % (mean, std_dev))
    img_to_show = img_stat > mean + (float(factor) * std_dev)
    show_one_img(img_to_show)
    hsv_processing_menu()


#  Ask for rgb filtering range
def prepare_rgb_range():
    print("Please insert RGB filtering range... (0-255) ")

    while True:
        try:
            r_min = int(input("R minimum: "))
            break
        except ValueError:
            print("Only numbers between 0 and 255...")

    while True:
        try:
            r_max = int(input("R maximum: "))
            break
        except ValueError:
            print("Only numbers between 0 and 255...")

    while True:
        try:
            g_min = int(input("G minimum: "))
            break
        except ValueError:
            print("Only numbers between 0 and 255...")

    while True:
        try:
            g_max = int(input("G maximum: "))
            break
        except ValueError:
            print("Only numbers between 0 and 255...")

    while True:
        try:
            b_min = int(input("B minimum: "))
            break
        except ValueError:
            print("Only numbers between 0 and 255...")

    while True:
        try:
            b_max = int(input("B maximum: "))
            break
        except ValueError:
            print("Only numbers between 0 and 255...")

    rgb_range = [(r_min, r_max), (g_min, g_max), (b_min, b_max)]
    return rgb_range


#  Filtering specific channel of HSV between values
def rgb_range_filter():
    img_rgb_filter = load_a_picture()
    arr = np.array(np.asarray(img_rgb_filter))

    # get_rgb = findrgbrange(img_rgb_filter)

    if len(arr.shape) >= 3:
        try:
            print("***Colour filtering***")
            # rgb_range = prepare_rgb_range()  #
            get_rgb = findrgbrange(img_rgb_filter)
            print(get_rgb)
            rgb_range = [(get_rgb[0], 255), (get_rgb[1], 255), (get_rgb[2], 255)]

            red_range = np.logical_and(rgb_range[0][0] <= arr[:, :, 0], arr[:, :, 0] <= rgb_range[0][1])
            # arr[np.logical_not(red_range), 0] = 0

            green_range = np.logical_and(rgb_range[1][0] <= arr[:, :, 1], arr[:, :, 1] <= rgb_range[1][1])

            blue_range = np.logical_and(rgb_range[2][0] <= arr[:, :, 2], arr[:, :, 2] <= rgb_range[2][1])

            redgreen = np.logical_or(red_range, green_range)
            redgreenblue = np.logical_or(redgreen, blue_range)

            arr[np.logical_not(redgreenblue), 0] = 0
            arr[np.logical_not(redgreenblue), 1] = 0
            arr[np.logical_not(redgreenblue), 2] = 0

            out_image = Image.fromarray(arr)

            # Save and show changes
            add_to_stack(out_image)
            show_img(actual_loaded)
            processing_menu()

        except ValueError:
            print(ValueError)
            print("Can't filter this image")
            input("press any key to back to processing menu...")
            processing_menu()


    else:
        try:
            print("***Grayscale filtering***")
            get_rgb = findrgbrange(img_rgb_filter)
            print(get_rgb)
            gray_range = np.logical_and(get_rgb[0] <= arr[:], arr[:] <= 255)

            arr[np.logical_not(gray_range)] = 0

            show_one_img(arr)

            out_image = Image.fromarray(arr)

            # Save and show changes
            add_to_stack(out_image)
            show_img(actual_loaded)
            processing_menu()

        except ValueError:
            print(ValueError)
            print("Can't filter this image")
            input("press any key to back to processing menu...")
            processing_menu()


# HSV range filter
def hsv_range_filter(img_stat, min_value, max_value):
    print("Filter between: %s %s" % (np.int(min_value), max_value))

    if min_value <= max_value:
        ret, img_to_show = cv2.threshold(img_stat, int(min_value), int(max_value), cv2.THRESH_TOZERO)
        add_to_stack(img_to_show)
        show_img(actual_loaded)
    else:
        print("Not implemented yet!")

    hsv_processing_menu()


#  Create and show histogram
def show_histogram():
    img_hist = load_a_picture()
    if len(img_hist.shape) >= 3:
        try:
            print("***Colour histogram***")
            for i, col in enumerate(['b', 'g', 'r']):
                hist = cv2.calcHist([img_hist], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
                plt.xlim([0, 256])

            plt.show()

        except ValueError:
            print(ValueError)
            print("Can't get the histogram of selected image")
            input("press any key to back to processing menu...")
            processing_menu()
    else:
        try:
            print("***Grayscale histogram***")
            # gray_image = cv2.cvtColor(img_hist, cv2.COLOR_BGR2GRAY)
            histogram = cv2.calcHist([img_hist], [0], None, [256], [0, 256])
            plt.plot(histogram, color='k')
            plt.show()
            processing_menu()

        except ValueError:
            print(ValueError)
            print("Can't get the histogram of selected image")
            input("press any key to back to processing menu...")
            processing_menu()


# Open image matrix in myavi
# def visualisation():
#    from mayavi import mlab
#    pts = mlab.points3d(imageContainer.image_matrix[0], imageContainer.image_matrix[1],
#                        imageContainer.image_matrix[2], imageContainer.image_matrix[2],
#                        scale_mode='none', scale_factor=0.2)

#    mesh = mlab.pipeline.delaunay2d(pts)
#    surf = mlab.pipeline.surface(mesh)
#    mlab.show()


# Call the menu function
if __name__ == "__main__":
    main_menu()
