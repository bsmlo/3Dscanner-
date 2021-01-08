# Different operations on images,
# control form main menu

import EditHandler
import imagecontainer as ic
import matplotlib.pyplot as plt

from numpy import save
from numpy import load

# Image container
imageContainer = ic.ImageContainer()


def visualisation():
    from mayavi import mlab

    pts = mlab.points3d(imageContainer.image_matrix[0], imageContainer.image_matrix[1],
                        imageContainer.image_matrix[2], imageContainer.image_matrix[2],
                        scale_mode='none', scale_factor=0.2)

    #    pts = mlab.points3d(imageContainer.image_matrix[0], imageContainer.image_matrix[1],
    #                        imageContainer.image_matrix[2], imageContainer.image_matrix[2],
    #                        scale_mode='none', scale_factor=0.2)

    # mesh = mlab.pipeline.delaunay2d(pts)
    # surf = mlab.pipeline.surface(mesh)

    mlab.show()


class Operations(EditHandler.EditHandler):
    def __init__(self):
        super().__init__()
        self.images_patch = 'E:/Data/Dokumenty/Studia/Praca/MGR/Serie/Test3-sub'

    def image_show(self):
        image = EditHandler.EditHandler.actual_image(self)
        plt.imshow(image)
        plt.show()

    # load_paths images from the path
    def load_images(self):
        # Image container test
        # Load all images from selected path
        if imageContainer.number_of_elements == 0:
            imageContainer.load_paths(r'%s' % (self.images_patch), r'.jpg')  ### loading off
            print('%s' % (self.images_patch))
            self.image_to_edit(int(imageContainer.number_of_elements / 2))  # load the middle image form the list
            print('%s' % imageContainer.number_of_elements)

        else:
            print(f"Images already loaded to the list... {imageContainer.number_of_elements} items loaded.")
            decision = input('Reload? Y/n')
            if decision == 'Y':
                imageContainer.load_paths(r'%s' % (self.images_patch), r'.jpg')
                self.image_to_edit(int(imageContainer.number_of_elements / 2))
                print('%s' % imageContainer.number_of_elements)

    def image_to_edit(self, index=-1):
        if index == -1:
            print('Select image')
            while True:
                try:
                    img_number = int(input('between 1 and %s' % imageContainer.number_of_elements))
                    if img_number < 1 or img_number > imageContainer.number_of_elements:
                        print('Out of range!')
                        continue
                    else:
                        EditHandler.EditHandler.get_new(self, imageContainer.image_list[img_number - 1])
                        break

                except ValueError:
                    print('Wrong input!')
                    continue
        else:
            EditHandler.EditHandler.get_new(self, imageContainer.image_list[index])

    # Edit in diferent threads
    def dosequenceThreads(self):
        imageContainer.CallSequenceThreads(imageContainer.image_list, imageContainer.resolution)

    # Save image matrix as a plane txt
    def save_matrix(self):
        matrix_path = '{}\{}'.format(imageContainer.folder_directory, 'img_mtx.npy')
        print(matrix_path)
        save(matrix_path, imageContainer.image_matrix)

    # Save image matrix as a plane txt
    def load_matrix(self):
        matrix_path = '{}\{}'.format(imageContainer.folder_directory, 'img_mtx.npy')
        print(matrix_path)
        imageContainer.image_matrix = load(matrix_path)
        print('Loaded')
        print(imageContainer.image_matrix.shape)
