
import numpy as np

def make_homogeneous(points):
    """
    Convert Cartesian to homogeneous coordinates.
    :param points: Nx2 numpy array of Cartesian coordinates
    :return: Nx3 numpy array of homogeneous coordinates
    """
    dim = points.shape[1]
    h_points = np.ones((points.shape[0], dim + 1), points.dtype)
    h_points[:, 0:dim] = np.copy(points)

    return h_points


def make_cartesian(h_points):
    """
    Convert homogeneous to Cartesian coordinates.
    :param h_points: Nx3 numpy array of homogeneous coordinates
    :return: Nx2 numpy array of Cartesian coordinates
    """
    dim = h_points.shape[1] - 1
    points = np.zeros((h_points.shape[0], dim), h_points.dtype)
    points[:, :] = h_points[:, :dim] / h_points[:, dim, None]

    return points


def make_homogeneous_point(point):
    dim = point.shape[0]
    h_point = np.ones((dim + 1,), point.dtype)
    h_point[0:dim] = np.copy(point)

    return h_point


def make_cartesian_point(h_point):
    dim = h_point.shape[0] - 1
    point = np.zeros((dim,), h_point.dtype)
    point[:] = h_point[:dim] / h_point[dim, None]

    return point

def point_distance(p0, p1):
    return np.linalg.norm(p0 - p1)

def point_is_left(line, p):
    p0, p1 = line
    return ((p1[0] - p0[0]) * (p[1] - p0[1]) - (p1[1] - p0[1]) * (p[0] - p0[0])) > 0