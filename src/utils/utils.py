from scipy.misc import imread
from scipy.misc import imresize

def split_data(ground_truths, validation_split=.2):
    ground_truth_keys = sorted(ground_truths.keys())
    training_split = 1 - validation_split
    num_train = int(round(training_split * len(ground_truth_keys)))
    train_keys = ground_truth_keys[:num_train]
    validation_keys = ground_truth_keys[num_train:]
    return train_keys, validation_keys


def get_class_names(dataset_name='VOC2007'):
    if dataset_name == 'VOC2007':
        class_names = ['background','aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                       'diningtable', 'dog', 'horse', 'motorbike', 'person',
                       'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    else:
        raise Exception('Invalid %s dataset' % dataset_name)
    return class_names

def scheduler(epoch, decay=0.9, base_learning_rate=3e-4):
    return base_learning_rate * decay**(epoch)

def get_arg_to_class(class_names):
    return dict(zip(list(range(len(class_names))), class_names))

def read_image(file_name):
    return imread(file_name)

def resize_image(image_array, size):
    return imresize(image_array, size)
