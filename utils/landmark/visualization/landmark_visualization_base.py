
import numpy as np

from utils.io.image import normalize_image_to_np_range
from utils.landmark.common import Landmark
from utils.sitk_image import label_to_rgb, resample_to_spacing
from utils.sitk_np import sitk_to_np_no_copy


class LandmarkVisualizationBase(object):
    """
    Class for landmark groundtruth and prediction visualization. Also performs axis projection for 3D images.
    """
    def __init__(self, radius=3, line_color=None, missed_color=None, too_many_color=None, flip_y_axis=True, normalize_image=True, annotations=None):
        self.radius = radius
        self.line_color = line_color or [0.0, 0.0, 0.0]
        self.missed_color = missed_color or [255.0, 0.0, 0.0]
        self.too_many_color = too_many_color or [0.0, 255.0, 0.0]
        self.flip_y_axis = flip_y_axis
        self.normalize_image = normalize_image
        self.annotations = annotations

    def prepare_image_canvas(self, image_np_list):
        """
        Prepares a canvas (e.g., np image or matplotlib axis) for each projection image.
        :param image_np_list: A list of np images, representing the image projections.
        :return: A list of image canvases. Will be given to merge_image_canvas.
        """
        raise NotImplementedError

    def merge_image_canvas(self, image_canvas_list):
        """
        Merge image canvases to a single image.
        :param image_canvas_list: A list of image canvas objects (returned by prepare_image_canvas).
        :return: An image canvas.
        """
        raise NotImplementedError

    def save(self, image_canvas_merged, filename):
        """
        Save the merged image canvas to the filename.
        :param image_canvas_merged: A merged image canvas (returned by merge_image_canvas).
        :param filename: The filename to save the image canvas to.
        """
        raise NotImplementedError

    def visualize_landmark(self, image_canvas, landmark, color, annotation, annotation_color):
        """
        Visualize a single landmark.
        :param image_canvas: The image canvas object.
        :param landmark: The landmark.
        :param color: The landmark color.
        :param annotation: The annotation string.
        :param annotation_color: The annotation color.
        """
        raise NotImplementedError

    def visualize_from_to_landmark_offset(self, image_canvas, landmark_from, landmark_to, color):
        """
        Visualize the offset between to landmarks.
        :param image_canvas: The image canvas object.
        :param landmark_from: The from landmark.
        :param landmark_to: The to landmark.
        :param color: The landmark color.
        """
        raise NotImplementedError

    def visualize_landmarks(self, image_canvas, predicted=None, groundtruth=None):
        """
        Visualize predicted and groundtruth landmarks to an image canvas.
        If both predicted and groundtruth landmarks are set, call visualize_landmark_pair, otherwise call visualize_landmark_single.
        :param image_canvas: The image canvas to write to.
        :param predicted: The list of predicted landmarks.
        :param groundtruth: The list of groundtruth landmarks.
        """
        if predicted is None and groundtruth is not None:
            for i, l in enumerate(groundtruth):
                self.visualize_landmark_single(image_canvas, l, i)
        elif predicted is not None and groundtruth is None:
            for i, l in enumerate(predicted):
                self.visualize_landmark_single(image_canvas, l, i)
        elif predicted is not None and groundtruth is not None:
            for i, (p, gt) in enumerate(zip(predicted, groundtruth)):
                self.visualize_landmark_pair(image_canvas, p, gt, i)

    def visualize_landmark_single(self, image_canvas, landmark, index):
        """
        Visualize a single landmark.
        :param image_canvas: The image canvas to write to.
        :param landmark: The landmark.
        :param index: The landmark index. Used for annotation and color.
        """
        color = label_to_rgb(index, float_range=False)
        annotation = None if self.annotations is None or index not in self.annotations else self.annotations[index]
        if landmark.is_valid:
            self.visualize_landmark(image_canvas, landmark, color, annotation, color)

    def visualize_landmark_pair(self, image_canvas, prediction, groundtruth, index):
        """
        Visualize a landmark pair.
        :param image_canvas: The image canvas to write to.
        :param prediction: The predicted landmark. Will be visualized.
        :param groundtruth: The groundtruth landmark. An offset vector from prediction to groundtruth will be visualized.
        :param index: The landmark index. Used for annotation and color.
        """
        annotation = None if self.annotations is None or index not in self.annotations else self.annotations[index]
        if prediction.is_valid and groundtruth.is_valid:
            color = label_to_rgb(index, float_range=False)
            self.visualize_landmark(image_canvas, prediction, color, annotation, color)
            self.visualize_from_to_landmark_offset(image_canvas, prediction, groundtruth, self.line_color)
        elif prediction.is_valid and not groundtruth.is_valid:
            color = self.too_many_color
            self.visualize_landmark(image_canvas, prediction, color, annotation, color)
        elif not prediction.is_valid and groundtruth.is_valid:
            color = self.missed_color
            self.visualize_landmark(image_canvas, groundtruth, color, annotation, color)

    def project_landmarks(self, landmarks, axis):
        """
        Project landmarks to an axis.
        :param landmarks: The landmarks list.
        :param axis: The axis to project to.
        :return: List of projected landmarks.
        """
        if landmarks is None:
            return None
        projected_landmarks = []
        for l in landmarks:
            if not l.is_valid:
                projected_landmarks.append(Landmark(is_valid=False))
            else:
                projected_landmarks.append(Landmark([l.coords[i] for i in range(len(l.coords)) if i != axis]))
        return projected_landmarks

    def visualize_projections(self, image_sitk, predicted=None, groundtruth=None, filename=None):
        """
        Visualize landmarks or landmark pairs onto projections of a given sitk image.
        If both predicted and groundtruth landmarks are set, visualize landmarks and offsets, otherwise visualize landmarks only.
        :param image_sitk: The sitk image (that will be projected in case of 3D).
        :param predicted: The list of predicted landmarks. May be None.
        :param groundtruth: The list of groundtruth landmarks. May be None.
        :param filename: The filename to save the image to.
        """
        dim = image_sitk.GetDimension()
        image_sitk = resample_to_spacing(image_sitk, [1.0] * dim)
        image_np = sitk_to_np_no_copy(image_sitk)
        if self.normalize_image:
            image_np = normalize_image_to_np_range(image_np, 'min_max', np.uint8)
        image_np = image_np.astype(np.uint8)
        if dim == 2:
            # do not project in case of 2D
            image_canvas_list = self.prepare_image_canvas([image_np])
            self.visualize_landmarks(image_np, predicted, groundtruth)
            image_canvas_merged = self.merge_image_canvas(image_canvas_list)
        else:
            # project to every axis in case of 3D
            image_canvas_list = self.prepare_image_canvas([np.max(image_np, axis=axis) for axis in [0, 1, 2]])
            for axis in [0, 1, 2]:
                image_canvas = image_canvas_list[axis]
                projected_predicted = self.project_landmarks(predicted, 2-axis)
                projected_groundtruth = self.project_landmarks(groundtruth, 2-axis)
                self.visualize_landmarks(image_canvas, projected_predicted, projected_groundtruth)
            image_canvas_merged = self.merge_image_canvas(image_canvas_list)
        self.save(image_canvas_merged, filename)
