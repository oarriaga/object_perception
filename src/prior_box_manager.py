import numpy as np

class PriorBoxManager(object):
    def __init__(self, prior_boxes, num_classes, background_id=0,
                                            overlap_threshold=.5):
        self.prior_boxes = prior_boxes
        self.num_priors = self.prior_boxes.shape[0]
        self.num_classes = num_classes
        self.overlap_threshold = overlap_threshold
        self.background_id = background_id

    def _calculate_intersection_over_unions(self, ground_truth_data):
        ground_truth_x_min = ground_truth_data[0]
        ground_truth_y_min = ground_truth_data[1]
        ground_truth_x_max = ground_truth_data[2]
        ground_truth_y_max = ground_truth_data[3]
        prior_boxes_x_min = self.prior_boxes[:, 0]
        prior_boxes_y_min = self.prior_boxes[:, 1]
        prior_boxes_x_max = self.prior_boxes[:, 2]
        prior_boxes_y_max = self.prior_boxes[:, 3]
        # calculating the intersection
        intersections_x_min = np.maximum(prior_boxes_x_min, ground_truth_x_min)
        intersections_y_min = np.maximum(prior_boxes_y_min, ground_truth_y_min)
        intersections_x_max = np.minimum(prior_boxes_x_max, ground_truth_x_max)
        intersections_y_max = np.minimum(prior_boxes_y_max, ground_truth_y_max)
        intersected_widths = intersections_x_max - intersections_x_min
        intersected_heights = intersections_y_max - intersections_y_min
        intersected_widths = np.maximum(intersected_widths, 0)
        intersected_heights = np.maximum(intersected_heights, 0)
        intersections = intersected_widths * intersected_heights
        # calculating the union
        prior_box_widths = prior_boxes_x_max - prior_boxes_x_min
        prior_box_heights = prior_boxes_y_max - prior_boxes_y_min
        prior_box_areas = prior_box_widths * prior_box_heights
        ground_truth_width = ground_truth_x_max - ground_truth_x_min
        ground_truth_height = ground_truth_y_max - ground_truth_y_min
        ground_truth_area = ground_truth_width * ground_truth_height
        unions = prior_box_areas + ground_truth_area - intersections
        intersection_over_unions = intersections / unions
        return intersection_over_unions

    def _assign_boxes_to_object(self, ground_truth_box, return_iou=True):
        ious = self._calculate_intersection_over_unions(ground_truth_box)
        assigned_boxes = np.zeros((self.num_priors, 4 + return_iou))
        assign_mask = ious > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[ious.argmax()] = True
        if return_iou:
            assigned_boxes[:, -1][assign_mask] = ious[assign_mask]
        assigned_to_object_boxes = self.prior_boxes[assign_mask]
        assigned_boxes[assign_mask, 0:4] = assigned_to_object_boxes
        return assigned_boxes.ravel()

    def assign_boxes(self, ground_truth_data):
        assignments = np.zeros((self.num_priors, 4 + self.num_classes + 8))
        assignments[:, 4 + self.background_id] = 1.0
        num_objects_in_image = len(ground_truth_data)
        if num_objects_in_image == 0:
            return assignments
        assigned_boxes = np.apply_along_axis(self._assign_boxes_to_object,
                                            1, ground_truth_data[:, :4])
        assigned_boxes = assigned_boxes.reshape(-1, self.num_priors, 5)
        best_iou = assigned_boxes[:, :, -1].max(axis=0)
        best_iou_indices = assigned_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_indices = best_iou_indices[best_iou_mask]
        num_assigned_boxes = len(best_iou_indices)
        assigned_boxes = assigned_boxes[:, best_iou_mask, :]
        assignments[best_iou_mask, :4] = assigned_boxes[best_iou_indices,
                                                np.arange(num_assigned_boxes),
                                                :4]
        assignments[:, 4][best_iou_mask] = 0
        assignments[:, 5:-8][best_iou_mask] = ground_truth_data[best_iou_indices, 5:]
        assignments[:, -8][best_iou_mask] = 1
        return assignments

if __name__ == "__main__":
    import random

    from prior_box_creator import PriorBoxCreator
    from XML_parser import XMLParser

    data_path = '../datasets/german_open_dataset/annotations/'
    data_manager = XMLParser(data_path)
    class_names = data_manager.class_names
    num_classes = len(class_names)
    print('Found classes: \n', class_names)
    ground_truth_data = data_manager.get_data()
    sampled_key =  random.choice(list(ground_truth_data.keys()))
    sampled_data = ground_truth_data[sampled_key]

    prior_box_creator = PriorBoxCreator(image_shape=(300, 300))
    prior_boxes = prior_box_creator.create_boxes()
    prior_box_manager = PriorBoxManager(prior_boxes, num_classes)
    assigned_boxes = prior_box_manager.assign_boxes(sampled_data)
    object_mask = assigned_boxes[:, 4] != 1
    object_data = assigned_boxes[object_mask]
    print('Number of box assigned to different objects:', len(object_data))
