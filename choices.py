
class Choices:
    def __init__(self):
        self.functions_list = []
        self.removed_list = []

    #  Add next function to the list
    def add_choice(self, function_name):
        self.functions_list.append(function_name)
        #print(self.functions_list)

    #  remove last coice
    def undo_choice(self):
        if len(self.functions_list) > 0:
            self.removed_list.append(self.functions_list[-1]) # Move last choice to removed list
            del self.functions_list[-1]
            print(self.functions_list)
        else:
            print('Nothing to remove!!!')



    # get back last function
    def redo_choice(self):
        if len(self.removed_list) > 0:
            self.functions_list.append(self.removed_list[-1]) # Move last choice to removed list
            del self.removed_list[-1]
            print(self.functions_list)
        else:
            print('Can not redo choice!!!')


    # Return the list of choices
    def get_list(self):
        return self.functions_list
