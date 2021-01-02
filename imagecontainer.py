import glob
import cv2
import ntpath
import numpy as np
import ImageFilters
#import Sequence
import re
from numpy import asarray
from numpy import save
from numpy import load


# image container, loads and counts image collection


class ImageContainer(ImageFilters.Sequence):

    def __init__(self):
        super().__init__()
        self.image_list = []
        self.compare_images = []  # images before motphological thinnig
        self.image_names = []
        self.image_paths = []
        self.layer_number = 0
        self.image_type = ''
        self.folder_directory = ''
        self.number_of_elements = 0
        self.resolution = 0.082142 #0.2005102 #0.1074   39.2cm  # Length between two layers of scanning item 0.2142857 #

    # Load images to the list
    def load(self, directory, extension):
        # save path and extension
        self.image_type = extension
        self.folder_directory = directory

        # append paths to the list
        self.image_paths = glob.glob(r'{}/*{}'.format(self.folder_directory, self.image_type))

        # sort the names incresing
        self.image_paths.sort(key=lambda f: int(re.sub('\D', '', f)))

        # append names list
        for path in self.image_paths:
            self.image_names.append(ntpath.basename(path))

        # load images
        for image in self.image_paths:
            img = cv2.imread(image)
            self.image_list.append(img)

        self.count_images()

    def dosequence(self, functions):
        ImageFilters.Sequence.CallSequence(self, self.image_list, functions, self.resolution)
        #print(self.image_matrix)

    # Count images
    def count_images(self):
        self.number_of_elements = len(self.image_list)

    # Save collection to the specific folder
    def save(self):
        print("Not implement")
        print(self.image_paths)
        print(self.image_names)

    # Save image matrix as a plane txt
    def save_matrix(self):
        matrix_path = '{}\{}'.format(self.folder_directory, 'img_mtx.npy')
        print(matrix_path)
        save(matrix_path, self.image_matrix)

    # Save image matrix as a plane txt
    def load_matrix(self):
        matrix_path = '{}\{}'.format(self.folder_directory, 'img_mtx.npy')
        print(matrix_path)
        self.image_matrix = load(matrix_path)
        print('Loaded')
        print(self.image_matrix.shape)
