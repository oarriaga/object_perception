from skimage.feature import hog
from utils import _imresize
from utils import _imread
import os

class HOGExtractor(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.orientations = 8
        self.pixels_per_cell = (16, 16)
        self.cells_per_block = (1, 1)
        self.negative_per_positive_ratio = 3
        self.visualize = True
        self.file_names = None

    def get_features(self):
        file_names = os.listdir(self.data_path)
        hog_features = []
        #object_class = []
        for file_name in file_names:
            image_array = self._imread(self.data_path +  file_name)
            hog_features.append(hog(image_array, self.orientation,
                            self.pixels_per_cell, self.cells_per_block))
        return hog_features
