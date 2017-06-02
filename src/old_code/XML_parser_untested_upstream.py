import numpy as np
import os
from xml.etree import ElementTree
from utils import get_labels

class XMLParser(object):
    """ Preprocess the VOC2007 xml annotations data.

    # Arguments
        data_path: Data path to VOC2007 annotations

    # Return
        data: Dictionary which keys correspond to the image names
        and values are numpy arrays of shape (num_objects, 4 + num_classes)
        num_objects refers to the number of objects in that specific image
    """

    def __init__(self, data_path, background_id=None, class_names=None, dataset_name=None,
                suffix='.jpg', use_bounding_boxes=False, use_classes=False):
        self.path_prefix = data_path
        self.background_id = background_id
        if class_names == None:
            self.arg_to_class = get_labels(dataset_name='german_open_2017')
            self.class_to_arg = {value: key for key, value
                             in self.arg_to_class.items()}
            self.class_names = list(self.class_to_arg.keys())
            self.suffix = suffix
        else:
            if background_id != None and background_id != -1:
                class_names.insert(background_id, 'background')
            elif background_id == -1:
                class_names.append('background')
            keys = np.arange(len(class_names))
            self.arg_to_class = dict(zip(keys, class_names))
            self.class_names = class_names

        #consider adding the suffix here as well
        #self.suffix = suffix
        self.data = dict()
        self.use_bounding_boxes = use_bounding_boxes
        self._preprocess_XML()

    def get_data(self):
        return self.data

    def _preprocess_XML(self):
        filenames = os.listdir(self.path_prefix)
        num_files = len(filenames)
        if num_files == 0:
            raise Exception('empty directory')
        else:
            print('Number of files founded in directory:', num_files)
        #print(self.class_names)
        for filename in filenames:
            tree = ElementTree.parse(self.path_prefix + filename)
            root = tree.getroot()
            bounding_boxes = []
            one_hot_classes = []
            size_tree = root.find('size')
            if self.use_bounding_boxes == False:
                width = float(size_tree.find('width').text)
                height = float(size_tree.find('height').text)
            else:
                width = 1.0
                height = 1.0
            for object_tree in root.findall('object'):
                class_name = object_tree.find('name').text
                if class_name in self.class_names:
                    one_hot_class = self._to_one_hot2(class_name)
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
            image_name = image_name + self.suffix
            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            if len(bounding_boxes.shape) == 1:
                image_data = np.expand_dims(image_data, axis=0)
            self.data[image_name] = image_data

    def _to_one_hot(self, name):
        num_classes = len(self.class_to_arg)
        one_hot_vector = [0] * num_classes
        class_arg = self.class_to_arg[name]
        one_hot_vector[class_arg] = 1
        return one_hot_vector

    def _to_one_hot2(self, name):
        #one_hot_vector = [0] * self.num_classes
        one_hot_vector = [0] * 7
        #food
        if name == 'cereal':
            one_hot_vector[0] = 1
        elif name == 'cornflakes':
            one_hot_vector[0] = 1
        elif name == 'paprika':
            one_hot_vector[0] = 1
        elif name == 'peas':
            one_hot_vector[0] = 1
        elif name == 'salt':
            one_hot_vector[0] = 1
        elif name == 'tomato_paste':
            one_hot_vector[0] = 1
        elif name == 'potato':
            one_hot_vector[0] = 1
        elif name == 'potato_soup':
            one_hot_vector[0] = 1
        elif name == 'bread':
            one_hot_vector[0] = 1
        elif name == 'crackers':
            one_hot_vector[0] = 1
        elif name == 'noodles':
            one_hot_vector[0] = 1

        # container
        elif name == 'coffeecup':
            one_hot_vector[1] = 1
        elif name == 'white_bowl':
            one_hot_vector[1] = 1
        elif name == 'plate':
            one_hot_vector[1] = 1
        elif name == 'red_bowl':
            one_hot_vector[1] = 1
        elif name == 'basket':
            one_hot_vector[1] = 1
        elif name == 'bag':
            one_hot_vector[1] = 1

        # snack    
        elif name == 'pringles':
            one_hot_vector[2] = 1
        elif name == 'chocolate_cookies':
            one_hot_vector[2] = 1
        elif name == 'party_cracker':
            one_hot_vector[2] = 1
        elif name == 'egg':
            one_hot_vector[2] = 1

        # cleaning_stuff 
        elif name == 'cloth':
            one_hot_vector[3] = 1
        elif name == 'paper':
            one_hot_vector[3] = 1
        elif name == 'sponge':
            one_hot_vector[3] = 1
        elif name == 'towel':
            one_hot_vector[3] = 1

        # drink
        elif name == 'coke':
            one_hot_vector[4] = 1
        elif name == 'water':
            one_hot_vector[4] = 1
        elif name == 'orange_drink':
            one_hot_vector[4] = 1
        elif name == 'banana_milk':
            one_hot_vector[4] = 1
        elif name == 'cappucino':
            one_hot_vector[4] = 1

        # fruit
        elif name == 'pear':
            one_hot_vector[5] = 1
        elif name == 'apple':
            one_hot_vector[5] = 1
        elif name == 'lemon':
            one_hot_vector[5] = 1

        # cutlery
        elif name == 'fork_spoon_knife':
            one_hot_vector[6] = 1
        else:
            print('unknown label: %s' %name)

        return one_hot_vector



if __name__ == '__main__':
    data_path = '../../datasets/VOCdevkit/VOC2007/Annotations/'
    classes = ['bottle', 'sofa', 'tvmonitor', 'diningtable', 'chair']
    xml_parser = XMLParser(data_path, background_id=None, class_names=classes)
    ground_truths = xml_parser.get_data()
    print(len(ground_truths.keys()))
    print(xml_parser.arg_to_class)

