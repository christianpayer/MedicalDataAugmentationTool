
import numpy as np

from utils.io.common import create_directories_for_file_name
from utils.io.image import write
from utils.landmark.visualization.landmark_visualization_base import LandmarkVisualizationBase
from utils.np_image import gallery, draw_circle, draw_line
from utils.sitk_np import np_to_sitk


class LandmarkVisualization(LandmarkVisualizationBase):
    """
    Class for landmark groundtruth and prediction visualization. Also performs axis projection for 3D images. Uses np for visualization.
    """
    def prepare_image_canvas(self, image_np_list):
        """
        Prepares a canvas (e.g., np image or matplotlib axis) for each projection image.
        :param image_np_list: A list of np images, representing the image projections.
        :return: A list of image canvases. Will be given to merge_image_canvas.
        """
        return [np.stack([image_np] * 3, axis=-1) for image_np in image_np_list]

    def merge_image_canvas(self, image_canvas_list):
        """
        Merge image canvases to a single image.
        :param image_canvas_list: A list of image canvas objects (returned by prepare_image_canvas).
        :return: An image canvas.
        """
        image_np_rgb_list = []
        for image_canvas in image_canvas_list:
            image_canvas = np.transpose(image_canvas, [2, 0, 1])
            if self.flip_y_axis:
                image_canvas = np.flip(image_canvas, 1)
            image_np_rgb_list.append(image_canvas)
        image_np_rgb_gallery = gallery(image_np_rgb_list, len(image_np_rgb_list))
        return image_np_rgb_gallery

    def save(self, image_canvas_merged, filename):
        """
        Save the merged image canvas to the filename.
        :param image_canvas_merged: A merged image canvas (returned by merge_image_canvas).
        :param filename: The filename to save the image canvas to.
        """
        create_directories_for_file_name(filename)
        image_canvas_merged_sitk = np_to_sitk(np.transpose(image_canvas_merged, [1, 2, 0]), is_vector=True)
        write(image_canvas_merged_sitk, filename, compress=True)

    def visualize_landmark(self, image_canvas, landmark, color, annotation, annotation_color):
        """
        Visualize a single landmark.
        :param image_canvas: The image canvas object.
        :param landmark: The landmark.
        :param color: The landmark color.
        :param annotation: The annotation string.
        :param annotation_color: The annotation color.
        """
        coords = landmark.coords
        if self.spacing is not None:
            coords = coords / self.spacing
        draw_circle(image_canvas, [coords[1], coords[0]], self.radius, color)

    def visualize_from_to_landmark_offset(self, image_canvas, landmark_from, landmark_to, color):
        """
        Visualize the offset between to landmarks.
        :param image_canvas: The image canvas object.
        :param landmark_from: The from landmark.
        :param landmark_to: The to landmark.
        :param color: The landmark color.
        """
        draw_line(image_canvas, [landmark_from.coords[1], landmark_from.coords[0]], [landmark_to.coords[1], landmark_to.coords[0]], color)
