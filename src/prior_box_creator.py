import numpy as np
import pickle

class PriorBoxCreator(object):
    def __init__(self, image_shape=(300, 300)):
        self.image_shape = image_shape
        self.model_configurations = self._get_model_configurations()

    def _get_model_configurations(self):
        return pickle.load(open('utils/model_configurations.p', 'rb'))

    def create_boxes(self):
        image_width, image_height = self.image_shape
        boxes_parameters = []
        for layer_config in self.model_configurations:
            layer_width = layer_config["layer_width"]
            layer_height = layer_config["layer_height"]
            num_priors = layer_config["num_prior"]
            aspect_ratios = layer_config["aspect_ratios"]
            min_size = layer_config["min_size"]
            max_size = layer_config["max_size"]
            step_x = 0.5 * (float(image_width) / float(layer_width))
            step_y = 0.5 * (float(image_height) / float(layer_height))
            linspace_x = np.linspace(step_x, image_width - step_x, layer_width)
            linspace_y = np.linspace(step_y, image_height - step_y, layer_height)
            centers_x, centers_y = np.meshgrid(linspace_x, linspace_y)
            centers_x = centers_x.reshape(-1, 1)
            centers_y = centers_y.reshape(-1, 1)
            assert(num_priors == len(aspect_ratios))
            prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
            prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors))

            box_widths = []
            box_heights = []
            for aspect_ratio in aspect_ratios:
                if aspect_ratio == 1 and len(box_widths) == 0:
                    box_widths.append(min_size)
                    box_heights.append(min_size)
                elif aspect_ratio == 1 and len(box_widths) > 0:
                    box_widths.append(np.sqrt(min_size * max_size))
                    box_heights.append(np.sqrt(min_size * max_size))
                elif aspect_ratio != 1:
                    box_widths.append(min_size * np.sqrt(aspect_ratio))
                    box_heights.append(min_size / np.sqrt(aspect_ratio))
            box_widths = 0.5 * np.array(box_widths)
            box_heights = 0.5 * np.array(box_heights)
            prior_boxes[:, ::4] -= box_widths
            prior_boxes[:, 1::4] -= box_heights
            prior_boxes[:, 2::4] += box_widths
            prior_boxes[:, 3::4] += box_heights
            prior_boxes[:, ::2] /= image_width
            prior_boxes[:, 1::2] /= image_height
            prior_boxes = prior_boxes.reshape(-1, 4)

            layer_prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)
            boxes_parameters.append(layer_prior_boxes)
        return np.concatenate(boxes_parameters, axis=0)

if __name__ == '__main__':
    prior_box_creator = PriorBoxCreator()
    prior_boxes = prior_box_creator.create_boxes()
    print('Number of prior boxes created:', len(prior_boxes))

