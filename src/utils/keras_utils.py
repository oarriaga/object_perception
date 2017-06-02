from keras.preprocessing import image as keras_image_preprocessor
from keras.applications.vgg16 import preprocess_input

def load_image(image_path, grayscale=False ,target_size=None):
    image = keras_image_preprocessor.load_img(image_path,
                                                grayscale ,
                                    target_size=target_size)
    return keras_image_preprocessor.img_to_array(image)

def preprocess_images(image_array):
    return preprocess_input(image_array)

def get_arg_to_class(class_names):
    return dict(zip(list(range(len(class_names))), class_names))


