import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from random import shuffle

def get_labels(dataset_name):
    if dataset_name == 'german_open_2017':
        return {0:'cereal',1:'coffeecup',2:'coke',3:'cornflakes',
                4:'party_cracker',5:'peas',6:'tomato_paste',
                7:'water', 8:'potato_soup', 9:'pringles',
                10:'salt',11:'crackers', 12:'lemon', 13:'orange_drink',
                14:'banana_milk', 15:'apple', 16:'cappucino', 17:'chocolate_cookies',
                18:'egg', 19:'paper', 20:'red_bowl', 21:'sponge', 22:'paprika',
                23:'pear', 24:'potato', 25:'bread', 26:'white_bowl', 27:'basket',
                28:'cloth', 29:'noodles', 30:'towel', 31:'plate', 32:'bag',
                33:'fork_spoon_knife'}
                #missing pepper
    else:
        raise Exception('Invalid dataset name')

def display_batch(batch_data):
    batch_size = len(batch_data)
    batch_images = batch_data[0]['iamge_array_input']
    batch_classes = batch_data[1]['predcitions']
    for image_arg in range(batch_size):
        image = batch_images[image_arg]
        label = np.argmax(batch_classes[image_arg])
        display_image(image, label)

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

def split_data(ground_truth_data, training_ratio=.8, do_shuffle=False):
    ground_truth_keys = sorted(ground_truth_data.keys())
    if do_shuffle == True:
        shuffle(ground_truth_keys)

    num_train = int(round(training_ratio * len(ground_truth_keys)))
    train_keys = ground_truth_keys[:num_train]
    validation_keys = ground_truth_keys[num_train:]
    return train_keys, validation_keys

def display_image(image_array, title=None):
    image_array =  np.squeeze(image_array).astype('uint8')
    plt.imshow(image_array)
    if title != None:
        plt.title(title)
    plt.show()

def to_categorical(integer_classes, num_classes=2):
    integer_classes = np.asarray(integer_classes, dtype='int')
    num_samples = integer_classes.shape[0]
    categorical = np.zeros((num_samples, num_classes))
    categorical[np.arange(num_samples), integer_classes] = 1
    return categorical
