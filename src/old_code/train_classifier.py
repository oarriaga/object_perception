from XML_parser import XMLParser
from keras.callbacks import ModelCheckpoint, CSVLogger
from image_generator import ImageGenerator
from models import simpler_CNN
from utils import split_data
from utils import get_labels

dataset_name = 'german_open_2017'
batch_size = 30
num_epochs = 250
input_shape = (48, 48, 3)
trained_models_path = '../trained_models/object_models/simpler_CNN'
ground_truth_path = '../datasets/german_open_dataset/annotations/'
images_path = '../datasets/german_open_dataset/images/'
labels = get_labels(dataset_name)
#num_classes = len(list(labels.keys()))
num_classes = 7
use_bounding_boxes = True

data_loader = XMLParser(ground_truth_path, dataset_name,
                        use_bounding_boxes=use_bounding_boxes)
ground_truth_data = data_loader.get_data()

train_keys, val_keys = split_data(ground_truth_data,
                                    training_ratio=.6,
                                    do_shuffle=True)

image_generator = ImageGenerator(ground_truth_data, batch_size, input_shape[:2],
                                train_keys, val_keys, None,
                                path_prefix=images_path,
                                vertical_flip_probability=0,
                                do_random_crop=False,
                                use_bounding_boxes=use_bounding_boxes)

model = simpler_CNN(input_shape, num_classes)
model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
print(model.summary())

print('Number classes: ', num_classes)
print('Classes:', labels)
print('Number of training samples:', len(train_keys))
print('Number of validation samples:', len(val_keys))

model_names = trained_models_path + '.{epoch:02d}-{val_loss:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=False)

csv_logger = CSVLogger('log_files/classifier_training.log')
model.fit_generator(image_generator.flow(mode='train'),
                    #steps_per_epoch=int(len(train_keys)/batch_size),
                    steps_per_epoch=200,
                    epochs=num_epochs, verbose=1,
                    callbacks=[csv_logger, model_checkpoint],
                    validation_data= image_generator.flow('val'),
                    validation_steps=int(len(val_keys)/batch_size))

