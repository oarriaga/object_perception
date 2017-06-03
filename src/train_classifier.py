from keras.callbacks import CSVLogger, ModelCheckpoint

from image_generator import ImageGenerator
from models import simple_CNN
from prior_box_creator import PriorBoxCreator
from prior_box_manager import PriorBoxManager
from utils.utils import split_data
from XML_parser import XMLParser

# parameters
batch_size = 5
num_epochs = 10000
image_shape=(48, 48, 3)
validation_split = .2
dataset_name = 'VOC2007'
dataset_root_path = '../datasets/' + dataset_name + '/'
annotations_path =  dataset_root_path + 'annotations/'
image_prefix = dataset_root_path + 'images/'
trained_models_path = '../trained_models/classification/simple_CNN'
log_file_path = 'classification.log'

# loading data
data_manager = XMLParser(annotations_path)
class_names = data_manager.class_names
num_classes = len(class_names)
print('Found classes: \n', class_names)
#ground_truth_data = data_manager.get_data()
ground_truth_data = data_manager.get_data(['background', 'bottle'])
print('Number of real samples:', len(ground_truth_data))

# creating prior boxes
prior_box_creator = PriorBoxCreator()
prior_boxes = prior_box_creator.create_boxes()
prior_box_manager = PriorBoxManager(prior_boxes, num_classes)

train_keys, val_keys = split_data(ground_truth_data, validation_split)
image_generator = ImageGenerator(ground_truth_data, prior_box_manager,
                                batch_size, image_shape[0:2], train_keys,
                                val_keys, image_prefix,
                                vertical_flip_probability=0,
                                suffix='')

# model parameters
model = simple_CNN(image_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
                                            metrics=['accuracy'])
model.summary()

# model callbacks
csv_logger = CSVLogger(log_file_path, append=False)
model_names = (trained_models_path +
                '.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5')
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False)

# model training
model.fit_generator(image_generator.flow(mode='train'),
                    steps_per_epoch=int(len(train_keys) / batch_size),
                    epochs=num_epochs, verbose=1,
                    #callbacks=[csv_logger, model_checkpoint],
                    validation_data= image_generator.flow('val'),
                    validation_steps=int(len(val_keys) / batch_size))
