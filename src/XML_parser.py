import os
from xml.etree import ElementTree
import numpy as np
from utils.utils import get_class_names

class XMLParser(object):
    """ Preprocess the any data annotated in XML format.

    # TODO: Add background label

    # Arguments
        data_path: Data path to XML annotations

    # Return
        data: Dictionary which keys correspond to the image names
        and values are numpy arrays of shape (num_objects, 4 + num_classes)
        num_objects refers to the number of objects in that specific image
        4 + num_classes correspond to the x_min, y_min, x_max and y_max
        bounding_boxes coordinates respectively followed by a one-hot-encoding
        of the class.
    """

    def __init__(self, data_path, class_names=None, dataset_name=None):
        self.path_prefix = data_path
        self.dataset_name = dataset_name
        self.class_names = class_names
        if self.class_names is None and self.dataset_name is None:
            self.class_names = self._find_class_names()
            self.class_names = ['background'] + self.class_names
        elif self.class_names is None and self.dataset_name is not None:
            self.class_names = get_class_names(self.dataset_name)

        self.num_classes = len(self.class_names)

        class_keys = np.arange(self.num_classes)
        self.arg_to_class = dict(zip(class_keys, self.class_names))
        self.class_to_arg = {value: key for key, value
                             in self.arg_to_class.items()}
        self._parse_XML_files()

    def _find_class_names(self):
        found_classes = []
        filenames = os.listdir(self.path_prefix)
        for filename in filenames:
            tree = ElementTree.parse(self.path_prefix + filename)
            root = tree.getroot()
            for object_tree in root.findall('object'):
                class_name = object_tree.find('name').text
                if class_name not in found_classes:
                    found_classes.append(class_name)
        return found_classes

    def get_data(self, class_names=None):
        if class_names is not None:
            self.class_names = class_names
            assert 'background' in self.class_names
            self.num_classes = len(self.class_names)
            class_keys = np.arange(self.num_classes)
            self.arg_to_class = dict(zip(class_keys, self.class_names))
            self.class_to_arg = {value: key for key, value
                             in self.arg_to_class.items()}
        return self._parse_XML_files()

    def _parse_XML_files(self):
        data = dict()
        filenames = os.listdir(self.path_prefix)
        for filename in filenames:
            tree = ElementTree.parse(self.path_prefix + filename)
            root = tree.getroot()
            bounding_boxes = []
            one_hot_classes = []
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            for object_tree in root.findall('object'):
                class_name = object_tree.find('name').text
                if class_name in self.class_names:
                    one_hot_class = self._to_one_hot(class_name)
                    one_hot_classes.append(one_hot_class)
                    for bounding_box in object_tree.iter('bndbox'):
                        xmin = float(bounding_box.find('xmin').text) / width
                        ymin = float(bounding_box.find('ymin').text) / height
                        xmax = float(bounding_box.find('xmax').text) / width
                        ymax = float(bounding_box.find('ymax').text) / height
                    bounding_box = [xmin, ymin, xmax, ymax]
                    bounding_boxes.append(bounding_box)
            if len(one_hot_classes) == 0:
                continue
            image_name = root.find('filename').text
            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            if len(bounding_boxes.shape) == 1:
                image_data = np.expand_dims(image_data, axis=0)
            data[image_name] = image_data
        return data

    def _to_one_hot(self, class_name):
        one_hot_vector = [0] * self.num_classes
        class_arg = self.class_to_arg[class_name]
        one_hot_vector[class_arg] = 1
        return one_hot_vector

if __name__ == '__main__':
    data_path = '../datasets/german_open_dataset/annotations/'
    data_manager = XMLParser(data_path)
    class_names = data_manager.class_names
    print('Found classes: \n', class_names)
    ground_truth_data = data_manager.get_data()
    print('Number of ground truth samples:', len(ground_truth_data))
    print('Using classes: \n', class_names[0:3])
    ground_truth_data = data_manager.get_data(class_names[0:3])
    print('Number of ground truth samples:', len(ground_truth_data))
