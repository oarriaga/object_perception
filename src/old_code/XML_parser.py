import numpy as np
import os
from xml.etree import ElementTree

from utils import get_class_to_arg

class XMLParser(object):
    def __init__(self, path_prefix, dataset_name ,suffix='.jpg'):
        self.path_prefix = path_prefix
        self.dataset_name = dataset_name
        self.class_to_arg = get_class_to_arg(self.dataset_name)
        self.class_names = list(self.class_to_arg.keys())
        self.suffix = suffix
        self.data = dict()
        self._preprocess_XML()

    def get_data(self):
        return self.data

    def _preprocess_XML(self):
        filenames = os.listdir(self.path_prefix)
        num_files = len(filenames)
        if num_files == 0:
           raise Exception('Empty directory')
        else:
            print('Number of files founded in directory:', num_files)
        for filename in filenames:
            tree = ElementTree.parse(self.path_prefix + filename)
            root = tree.getroot()
            bounding_boxes = []
            one_hot_classes = []
            for object_tree in root.findall('object'):
                class_name = object_tree.find('name').text
                if class_name in self.class_names:
                    one_hot_class = self._to_one_hot(class_name)
                    one_hot_classes.append(one_hot_class)
                    for bounding_box in object_tree.iter('bndbox'):
                        xmin = float(bounding_box.find('xmin').text)
                        ymin = float(bounding_box.find('ymin').text)
                        xmax = float(bounding_box.find('xmax').text)
                        ymax = float(bounding_box.find('ymax').text)
                    bounding_box = [xmin, ymin, xmax, ymax]
                    bounding_boxes.append(bounding_box)
            # if no class found go to the next image
            if len(one_hot_classes) == 0:
                continue
            image_name = root.find('filename').text
            image_name = image_name + self.suffix
            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            # same dimensions if you have a single box in the image
            if len(bounding_boxes.shape) == 1:
                image_data = np.expand_dims(image_data, axis=0)
            self.data[image_name] = image_data

    def _to_one_hot(self, name):
        num_classes = len(self.class_to_arg)
        one_hot_vector = [0] * num_classes
        class_arg = self.class_to_arg[name]
        one_hot_vector[class_arg] = 1
        return one_hot_vector
