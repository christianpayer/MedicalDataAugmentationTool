
import numpy as np

from utils.io.image import normalize_image_to_np_range
from utils.landmark.common import Landmark
from utils.sitk_image import label_to_rgb, resample_to_spacing
from utils.sitk_np import sitk_to_np_no_copy


class LandmarkVisualizationBase(object):
    """
    Class for landmark groundtruth and prediction visualization. Also performs axis projection for 3D images.
    """
    def __init__(self, dim=3,
                 projection_axes=None,
                 radius=3,
                 landmark_colors=None,
                 line_color=(0, 0, 0),
                 missed_color=(255, 0, 0),
                 too_many_color=(0, 255, 0),
                 flip_y_axis=None,
                 normalize_image=True,
                 annotations=None,
                 spacing=None):
        """
        Initializer.
        :param dim: The dim of the images.
        :param projection_axes: The projection axes to used. If None, use [0] for 2D and [0, 1, 2] for 3D.
        :param radius: The radius of the landmark points.
        :param landmark_colors: The landmark colors. If None, use label_to_rgb.
        :param line_color: The line color for visualizing offsets.
        :param missed_color: The color of landmarks that are in the groundtruth, but not predicted.
        :param too_many_color: The color of landmarks that are predicted, but not in the groundtruth.
        :param flip_y_axis: If True, flip y axis. If None, use True for 3D and False for 2D.
        :param normalize_image: If True, normalize intensity range of image.
        :param annotations: Dictionary of annotations, or None.
        :param spacing: The spacing to resample the image to. If None, use uniform spacing.
        """
        self.dim = dim
        self.projection_axes = projection_axes
        if self.projection_axes is None:
            self.projection_axes = [0] if self.dim == 2 else [0, 1, 2]
        self.radius = radius
        self.landmark_colors = landmark_colors
        self.line_color = line_color
        self.missed_color = missed_color
        self.too_many_color = too_many_color
        self.flip_y_axis = flip_y_axis or False if self.dim == 2 else True
        self.normalize_image = normalize_image
        self.annotations = annotations
        self.spacing = spacing or [1.0] * self.dim

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

    def prepare_image_canvas_list(self, image_sitk):
        """
        Prepare the image canvas list for the projections_axes.
        :param image_sitk: The image to project.
        :return: List of projected image canvases.
        """
        image_sitk = resample_to_spacing(image_sitk, self.spacing)
        image_np = sitk_to_np_no_copy(image_sitk)
        if self.normalize_image:
            image_np = normalize_image_to_np_range(image_np, 'min_max', np.uint8)
        image_np = image_np.astype(np.uint8)
        if self.dim == 2:
            return self.prepare_image_canvas([image_np])
        else:
            return self.prepare_image_canvas([np.max(image_np, axis=axis) for axis in self.projection_axes])

    def project_landmarks(self, landmarks, axis):
        """
        Project landmarks to an axis.
        :param landmarks: The landmarks list.
        :param axis: The axis to project to.
        :return: List of projected landmarks.
        """
        projected_landmarks = []
        for l in landmarks:
            if not l.is_valid:
                projected_landmarks.append(Landmark(is_valid=False))
            else:
                projected_landmarks.append(Landmark([l.coords[i] for i in range(len(l.coords)) if i != axis]))
        return projected_landmarks

    def project_landmarks_list(self, landmarks):
        """
        Projects landmarks to the projection_axes and returns a list of list of projected landmarks.
        :param landmarks: The landmarks.
        :return: List of list of projected landmarks.
        """
        if self.dim == 2:
            return [landmarks]
        else:
            return [self.project_landmarks(landmarks, 2 - axis) for axis in self.projection_axes]

    def landmark_color_for_index(self, landmark_colors, index):
        """
        Return landmark color for a given index. If landmark_colors is a list, use color at list index.
        Else if landmark_colors is not None, use landmark_colors, else use function label_to_rgb()
        :param landmark_colors: List of landmark_colors or landmark_color or None.
        :param index: The landmark index.
        :return: RGB tuple of landmark color.
        """
        if isinstance(landmark_colors, list):
            return landmark_colors[index]
        elif landmark_colors is not None:
            return landmark_colors
        else:
            return label_to_rgb(index, float_range=False)

    def visualize_landmarks(self, image_canvas, landmarks, landmark_colors, annotations):
        """
        Visualize landmarks to an image canvas.
        :param image_canvas: The image canvas to write to.
        :param landmarks: The list of landmarks.
        :param landmark_colors: The list of landmark_colors. If None, use function label_to_rgb.
        :param annotations: The annotations per landmark. If None, no annotation is used.
        """
        for i, l in enumerate(landmarks):
            landmark_color = self.landmark_color_for_index(landmark_colors, i)
            annotation = annotations[i] if annotations is not None else None
            self.visualize_landmark_single(image_canvas, l, landmark_color, annotation)

    def visualize_landmark_offsets(self, image_canvas, predicted, groundtruth, landmark_colors, annotations):
        """
        Visualize predicted and groundtruth landmarks to an image canvas.
        :param image_canvas: The image canvas to write to.
        :param predicted: The list of predicted landmarks.
        :param groundtruth: The list of groundtruth landmarks.
        :param landmark_colors: The list of landmark_colors. If None, use function label_to_rgb.
        :param annotations: The annotations per landmark. If None, no annotation is used.
        """
        for i, (p, gt) in enumerate(zip(predicted, groundtruth)):
            landmark_color = self.landmark_color_for_index(landmark_colors, i)
            annotation = annotations[i] if annotations is not None else None
            self.visualize_landmark_offset(image_canvas, p, gt, landmark_color, annotation)

    def visualize_landmark_single(self, image_canvas, landmark, landmark_color, annotation):
        """
        Visualize a single landmark.
        :param image_canvas: The image canvas to write to.
        :param landmark: The landmark.
        :param landmark_color: The landmark color.
        :param annotation: The annotation. May be none.
        """
        if landmark.is_valid:
            self.visualize_landmark(image_canvas, landmark, landmark_color, annotation, landmark_color)

    def visualize_landmark_offset(self, image_canvas, prediction, groundtruth, landmark_color, annotation):
        """
        Visualize a landmark pair.
        :param image_canvas: The image canvas to write to.
        :param prediction: The predicted landmark. Will be visualized.
        :param groundtruth: The groundtruth landmark. An offset vector from prediction to groundtruth will be visualized.
        :param landmark_color: The landmark color.
        :param annotation: The annotation. May be none.
        """
        if prediction.is_valid and groundtruth.is_valid:
            color = landmark_color
            self.visualize_landmark(image_canvas, prediction, color, annotation, color)
            self.visualize_from_to_landmark_offset(image_canvas, prediction, groundtruth, self.line_color)
        elif prediction.is_valid and not groundtruth.is_valid:
            color = self.too_many_color
            self.visualize_landmark(image_canvas, prediction, color, annotation, color)
        elif not prediction.is_valid and groundtruth.is_valid:
            color = self.missed_color
            self.visualize_landmark(image_canvas, groundtruth, color, annotation, color)

    def visualize_landmark_projections(self, image_sitk, landmarks, filename):
        """
        Visualize landmarks onto projections of a given sitk image.
        :param image_sitk: The sitk image (that will be projected in case of 3D).
        :param landmarks: The list of landmarks.
        :param filename: The filename to save the image to.
        """
        image_canvas_list = self.prepare_image_canvas_list(image_sitk)
        projected_landmarks_list = self.project_landmarks_list(landmarks)
        for image_canvas, projected_landmarks in zip(image_canvas_list, projected_landmarks_list):
            self.visualize_landmarks(image_canvas, projected_landmarks, self.landmark_colors, self.annotations)
        image_canvas_merged = self.merge_image_canvas(image_canvas_list)
        self.save(image_canvas_merged, filename)

    def visualize_prediction_groundtruth_projections(self, image_sitk, predicted, groundtruth, filename):
        """
        Visualize prediction groundtruth pairs onto projections of a given sitk image.
        :param image_sitk: The sitk image (that will be projected in case of 3D).
        :param predicted: The list of predicted landmarks.
        :param groundtruth: The list of groundtruth landmarks.
        :param filename: The filename to save the image to.
        """
        image_canvas_list = self.prepare_image_canvas_list(image_sitk)
        projected_predicted_list = self.project_landmarks_list(predicted)
        projected_groundtruth_list = self.project_landmarks_list(groundtruth)
        for image_canvas, projected_predicted, projected_groundtruth in zip(image_canvas_list, projected_predicted_list, projected_groundtruth_list):
            self.visualize_landmark_offsets(image_canvas, projected_predicted, projected_groundtruth, self.landmark_colors, self.annotations)
        image_canvas_merged = self.merge_image_canvas(image_canvas_list)
        self.save(image_canvas_merged, filename)

    def visualize_landmark_list_projections(self, image_sitk, landmarks_list, landmark_colors_list, filename):
        """
        Visualize list of landmarks onto projections of a given sitk image.
        :param image_sitk: The sitk image (that will be projected in case of 3D).
        :param landmarks_list: List of list of predicted landmarks.
        :param landmark_colors_list: List of list of landmark colors for each entry ofr landmarks_list. If None, use self.landmark_colors.
        :param filename: The filename to save the image to.
        """
        image_canvas_list = self.prepare_image_canvas_list(image_sitk)
        for i, landmarks in enumerate(landmarks_list):
            landmark_colors = self.landmark_colors if landmark_colors_list is None else landmark_colors_list[i]
            projected_landmarks_list = self.project_landmarks_list(landmarks)
            for image_canvas, projected_landmarks in zip(image_canvas_list, projected_landmarks_list):
                self.visualize_landmarks(image_canvas, projected_landmarks, landmark_colors, self.annotations)
        image_canvas_merged = self.merge_image_canvas(image_canvas_list)
        self.save(image_canvas_merged, filename)

    def visualize_offsets_to_reference_projections(self, image_sitk, reference_groundtruth, predicted_per_image_id_list, groundtruth_per_image_id, landmark_colors_list, filename):
        """
        Visualize landmarks or landmark pairs onto projections of a given sitk image.
        :param image_sitk: The sitk image (that will be projected in case of 3D).
        :param reference_groundtruth: The reference_groundtruth for the image.
        :param predicted_per_image_id_list: The list of dictionaries of predicted landmarks.
        :param groundtruth_per_image_id: The dictionary of groundtruth landmarks.
        :param landmark_colors_list: List of list of landmark colors for each entry ofr landmarks_list. If None, use self.landmark_colors.
        :param filename: The filename to save the image to.
        """
        image_canvas_list = self.prepare_image_canvas_list(image_sitk)
        for image_id, groundtruth in groundtruth_per_image_id.items():
            for i, predicted_per_image_id in enumerate(predicted_per_image_id_list):
                landmark_colors = None if landmark_colors_list is None else landmark_colors_list[i]
                offsets = [Landmark(p.coords - g.coords + r.coords) for p, g, r in zip(predicted_per_image_id[image_id], groundtruth, reference_groundtruth)]
                projected_offset_list = self.project_landmarks_list(offsets)
                for image_canvas, projected_offsets in zip(image_canvas_list, projected_offset_list):
                    self.visualize_landmarks(image_canvas, projected_offsets, landmark_colors, self.annotations)
        # visualize black dots on original groundtruth
        projected_reference_groundtruth_list = self.project_landmarks_list(reference_groundtruth)
        for image_canvas, projected_reference_groundtruth in zip(image_canvas_list, projected_reference_groundtruth_list):
            self.visualize_landmarks(image_canvas, projected_reference_groundtruth, [(0, 0, 0) for _ in range(len(projected_reference_groundtruth))], self.annotations)
        image_canvas_merged = self.merge_image_canvas(image_canvas_list)
        self.save(image_canvas_merged, filename)
