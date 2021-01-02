#  This class handling the list of images editing for test (one from the loaded list)

import numpy as np


class EditHandler:
    def __init__(self):
        self.image_editing_list = []  # Stores editing history of one image
        self.actual_loaded = 0  # Index of the actual loaded image

    def actual_image(self):
        image = self.image_editing_list[self.actual_loaded]
        return image

    def orginal_image(self):
        image = self.image_editing_list[0]
        return image

    #  Decreasing the index of loaded image
    def undo_edit(self):
        if self.actual_loaded > 0:
            # choices.undo_choice()  # remove last function
            self.actual_loaded = self.actual_loaded - 1

        else:
            print("Can't undo.")

        print(len(self.image_editing_list))

    #  Increasing the index of loaded image
    def redo_edit(self):
        if 0 <= self.actual_loaded < len(self.image_editing_list) - 1:
            # choices.redo_choice()  # Readd last deleted function to the list
            self.actual_loaded = self.actual_loaded + 1
            # self.actual_image()
        else:
            print("Can't redo.")

        print(len(self.image_editing_list))

    #  Adding to the stack of images
    def add_to_stack(self, image_to_add):
        print(len(self.image_editing_list))

        # Convert image to np ndarray class
        img_array_to_add = np.asarray(image_to_add)
        if self.actual_loaded + 1 == len(self.image_editing_list) or len(self.image_editing_list) == 1:
            self.image_editing_list.append(img_array_to_add)
            self.actual_loaded = self.actual_loaded + 1
            print(len(self.image_editing_list))

        #  If not, edited image will be add as next to loaded, all next will be deleted
        else:
            self.image_editing_list[self.actual_loaded + 1] = img_array_to_add
            self.actual_loaded = self.actual_loaded + 1
            len_of_img_stack = len(self.image_editing_list)
            if self.actual_loaded + 1 < len_of_img_stack:
                del self.image_editing_list[-(len_of_img_stack - (self.actual_loaded + 1)):]
            print(len(self.image_editing_list))

    #  Reset settings when new image is loaded
    def get_new(self, image):
        self.image_editing_list = []
        self.image_editing_list.append(image)
        self.actual_loaded = 0
