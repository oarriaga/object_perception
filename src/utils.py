import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize


def get_labels(dataset_name):
    if dataset_name == 'german_open_2017':
        return {0:'cereal',1:'coffeecup',2:'coke',3:'cornflakes',
                4:'party_cracker',5:'peas',6:'tomato_paste',
                7:'water', 8:'potato_soup', 9:'pringles',
                10:'salt',11:'crackers', 12:'lemon', 13:'orange_drink',
                14:'banana_milk'}
    else:
        raise Exception('Invalid dataset name')

def preprocess_input(images):
    """ preprocess input image to the CNN
    # Arguments: images or image of any shape
    """
    images = images/255.0
    return images

def _imread(image_name):
        return imread(image_name)

def _imresize(image_array, size):
        return imresize(image_array, size)

def split_data(ground_truth_data, training_ratio=.8):
    ground_truth_keys = sorted(ground_truth_data.keys())
    num_train = int(round(training_ratio * len(ground_truth_keys)))
    train_keys = ground_truth_keys[:num_train]
    validation_keys = ground_truth_keys[num_train:]
    return train_keys, validation_keys

def display_image(image_array):
    image_array =  np.squeeze(image_array).astype('uint8')
    plt.imshow(image_array)
    plt.show()

def to_categorical(integer_classes, num_classes=2):
    integer_classes = np.asarray(integer_classes, dtype='int')
    num_samples = integer_classes.shape[0]
    categorical = np.zeros((num_samples, num_classes))
    categorical[np.arange(num_samples), integer_classes] = 1
    return categorical
