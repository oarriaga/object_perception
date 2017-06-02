import numpy as np
from random import shuffle

from utils.keras_utils import preprocess_images
from utils.utils import resize_image
from utils.utils import read_image


class ImageGenerator(object):
    """ Image generator with saturation, brightness, lighting, contrast,
    horizontal flip and vertical flip transformations. It supports
    bounding boxes coordinates.

    TODO:
        - Finish preprocess_images method.
        - Add random crop method.
        - Finish support for not using bounding_boxes.
    """
    def __init__(self, ground_truth_data, box_manager, batch_size, image_size,
                train_keys, validation_keys, path_prefix=None,
                negative_positive_ratio = 3,
                saturation_var=0.5,
                brightness_var=0.5,
                contrast_var=0.5,
                lighting_std=0.5,
                horizontal_flip_probability=0.5,
                vertical_flip_probability=0.5,
                do_crop=True,
                crop_area_range=[0.75, 1.0],
                aspect_ratio_range=[3./4., 4./3.]):

        self.ground_truth_data = ground_truth_data
        self.box_manager = box_manager
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.train_keys = train_keys
        self.validation_keys = validation_keys
        self.image_size = image_size
        self.color_jitter = []
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.lighting_std = lighting_std
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability
        self.do_crop = do_crop
        self.crop_area_range = crop_area_range
        self.aspect_ratio_range = aspect_ratio_range

    def _gray_scale(self, image_array):
        return image_array.dot([0.299, 0.587, 0.114])

    def saturation(self, image_array):
        gray_scale = self._gray_scale(image_array)
        alpha = 2.0 * np.random.random() * self.brightness_var
        alpha = alpha + 1 - self.saturation_var
        image_array = alpha * image_array + (1 - alpha) * gray_scale[:, :, None]
        return np.clip(image_array, 0, 255)

    def brightness(self, image_array):
        alpha = 2 * np.random.random() * self.brightness_var
        alpha = alpha + 1 - self.saturation_var
        image_array = alpha * image_array
        return np.clip(image_array, 0, 255)

    def contrast(self, image_array):
        gray_scale = (self._gray_scale(image_array).mean() *
                        np.ones_like(image_array))
        alpha = 2 * np.random.random() * self.contrast_var
        alpha = alpha + 1 - self.contrast_var
        image_array = image_array * alpha + (1 - alpha) * gray_scale
        return np.clip(image_array, 0, 255)

    def lighting(self, image_array):
        covariance_matrix = np.cov(image_array.reshape(-1,3) /
                                    255.0, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigen_vectors.dot(eigen_values * noise) * 255
        image_array = image_array + noise
        return np.clip(image_array, 0 , 255)

    def horizontal_flip(self, image_array, box_corners):
        if np.random.random() < self.horizontal_flip_probability:
            image_array = image_array[:, ::-1]
            box_corners[:, [0, 2]] = 1 - box_corners[:, [2, 0]]
        return image_array, box_corners

    def vertical_flip(self, image_array, box_corners):
        if (np.random.random() < self.vertical_flip_probability):
            image_array = image_array[::-1]
            box_corners[:, [1, 3]] = 1 - box_corners[:, [3, 1]]
        return image_array, box_corners

    def transform(self, image_array, box_corners):
        shuffle(self.color_jitter)
        for jitter in self.color_jitter:
            image_array = jitter(image_array)

        if self.lighting_std:
            image_array = self.lighting(image_array)

        if self.horizontal_flip_probability > 0:
            image_array, box_corners = self.horizontal_flip(image_array,
                                                            box_corners)

        if self.vertical_flip_probability > 0:
            image_array, box_corners = self.vertical_flip(image_array,
                                                            box_corners)

        return image_array, box_corners

    def _denormalize_box(self, box_coordinates, original_image_size):
        original_image_width, original_image_height = original_image_size
        box_coordinates[:, 0] = box_coordinates[:, 0] * original_image_width
        box_coordinates[:, 1] = box_coordinates[:, 1] * original_image_height
        box_coordinates[:, 2] = box_coordinates[:, 2] * original_image_width
        box_coordinates[:, 3] = box_coordinates[:, 3] * original_image_height
        return box_coordinates

    def _mangage_keys(self, mode):
        if mode =='train':
            shuffle(self.train_keys)
            keys = self.train_keys
        elif mode == 'val' or  mode == 'demo':
            shuffle(self.validation_keys)
            keys = self.validation_keys
        return keys

    def _select_negative_samples(self, assigned_data):
        object_mask = assigned_data[:, 4] != 1
        object_data = assigned_data[object_mask]
        num_assigned_boxes = len(object_data)
        background_mask = np.logical_not(object_mask)
        background_data = assigned_data[background_mask]
        num_background_boxes = len(background_data)
        random_args = np.random.permutation(num_background_boxes)
        num_negative_boxes = self.positive_negative_ratio * num_assigned_boxes
        random_args = np.unravel_index(random_args[:num_negative_boxes],
                                        dims=len(background_data))
        background_data = background_data[random_args]
        return background_data, object_data

    def _crop_bounding_boxes(self, image_array, assigned_data):
        data = self._select_negative_samples(assigned_data)
        data = np.concatenate(data, axis=0)
        images = []
        classes = []
        for object_arg in range(len(data)):
            object_data = data[object_arg]
            image_array = self._crop_bounding_box(image_array, object_data)
            image_array = resize_image(image_array, self.image_size)
            classes.append(data[object_arg][4:])
        return images, classes

    def _crop_bounding_box(self,image_array, box_data):
        x_min = box_data[:, 0]
        y_min = box_data[:, 1]
        x_max = box_data[:, 2]
        y_max = box_data[:, 3]
        cropped_array = image_array[x_min:x_max, y_min:y_max].copy()
        return cropped_array

    def flow(self, mode='train'):
            while True:
                keys = self._mangage_keys(mode)
                inputs = []
                targets = []
                for key in keys:
                    image_path = self.path_prefix + key
                    image_array = read_image(image_path)
                    original_image_size = image_array.shape[:2]
                    #image_array = resize_image(image_array, self.image_size)
                    box_data = self.ground_truth_data[key].copy()
                    if mode == 'train' or mode == 'demo':
                        image_array, box_data = self.transform(image_array,
                                                                  box_data)
                    assigned_data = self.box_manager.assign_boxes(box_data)
                    assigned_data = self._denormalize_box(assigned_data,
                                                    original_image_size)
                    image_arrays, classes = self._crop_bounding_boxes(
                                            image_array, assigned_data)
                    inputs = inputs + image_arrays
                    targets = targets + classes
                    #inputs.append(image_array)
                    #targets.append(box_data)
                    if len(targets) == self.batch_size:
                        inputs = np.asarray(inputs)
                        targets = np.asarray(targets)
                        if mode == 'train' or mode == 'val':
                            inputs = preprocess_images(inputs)
                            yield self._wrap_in_dictionary(inputs, targets)
                        if mode == 'demo':
                            yield self._wrap_in_dictionary(inputs, targets)
                        inputs = []
                        targets = []

    def _wrap_in_dictionary(self, image_array, targets):
        return [{'input_1':image_array},
                {'predictions':targets}]
