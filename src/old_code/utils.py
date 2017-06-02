from scipy.misc import imread
from scipy.misc import imresize

def preprocess_input(images):
    images = images/255.0
    return images

def _imread(image_name):
        return imread(image_name)

def _imresize(image_array, size):
        return imresize(image_array, size)

def get_class_to_arg(dataset_name):
    if dataset_name == 'german_open_2017':
        # 0 means fruit, 1 container
        return {'apple':0, 'pear':0, 'paper_bag':1}
