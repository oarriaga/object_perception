from XML_parser import XMLParser
from image_generator import ImageGenerator
from utils import split_data
from utils import get_labels
from utils import display_image
import numpy as np

dataset_name = 'german_open_2017'
batch_size = 10
num_epochs = 30
input_shape = (48, 48, 3)
trained_models_path = '../trained_models/object_models/simpler_CNN'
ground_truth_path = '../datasets/german_open_dataset/annotations/'
images_path = '../datasets/german_open_dataset/images/'
labels = get_labels(dataset_name)
num_classes = len(list(labels.keys()))
use_bounding_boxes = True

data_loader = XMLParser(ground_truth_path, dataset_name,
                        use_bounding_boxes=use_bounding_boxes)
ground_truth_data = data_loader.get_data()

train_keys, val_keys = split_data(ground_truth_data,
                                    training_ratio=.5,
                                    do_shuffle=True)

image_generator = ImageGenerator(ground_truth_data, batch_size, input_shape[:2],
                                train_keys, val_keys, None,
                                path_prefix=images_path,
                                vertical_flip_probability=0,
                                do_random_crop=True,
                                use_bounding_boxes=use_bounding_boxes)

batch_data = next(image_generator.flow('demo'))

batch_images = batch_data[0]['image_array_input']
batch_classes = batch_data[1]['predictions']

for image_arg in range(batch_size):
    image = batch_images[image_arg]
    label_arg = np.argmax(batch_classes[image_arg])
    label = labels[label_arg]
    display_image(image, label)
