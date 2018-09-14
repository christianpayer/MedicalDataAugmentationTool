
from utils.random import float_uniform
import numpy as np
import math
import tensorflow as tf


class Plane(object):
    def __init__(self, first_axis, second_axis, theta):
        """
        Constructs a 2D rotation plane around the specified axes.
        :param first_axis:
        :param second_axis:
        """
        # The first axis defining the rotation plane.
        self.first_axis = first_axis
        # The second axis defining the rotation plane.
        self.second_axis = second_axis
        # The rotation angle in radians; or {@code NaN} if no rotation is assigned.
        self.theta = theta


class RotationMatrixBuilderNp(object):
    def __init__(self, dim):
        """
        Constructs a rotation matrix builder for the given dimension.
        :param dim:
        """
        # The dimension of rotation matrices produced by this builder.
        self.dim = dim
        # The 2D rotation planes currently used by this builder.
        self.planes = []

    def rotate_plane(self, i, j, theta):
        """
         Adds a rotation around the 2D rotation plane defined by the two axes.
          The plane is initially unrotated, but can be assigned a specific rotation
          angle if followed by setting theta.
        :param i:
        :param j:
        :return:
        """
        assert (i >= 0) and (i < self.dim) and (j >= 0) and (j < self.dim) and (i != j), 'invalid plane'
        self.planes.append(Plane(i, j, theta))

    def rotate_all_random(self):
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                plane = Plane(i, j, float_uniform(0, 2 * math.pi))
                self.planes.append(plane)

    def create(self):
        rotation = np.eye(self.dim)

        for plane in self.planes:
            rotation = np.matmul(rotation, self.rotation_matrix(plane))

        return rotation

    def rotation_matrix(self, plane):
        i = plane.first_axis
        j = plane.second_axis
        theta = plane.theta

        rotation = np.eye(self.dim)
        rotation[i, i] = np.cos(theta)
        rotation[i, j] = -np.sin(theta)
        rotation[j, i] = np.sin(theta)
        rotation[j, j] = np.cos(theta)

        return rotation


#def own_sparse_matmul(sp_a, sp_b):


class RotationMatrixBuilderTf(object):
    def __init__(self, dim):
        """
        Constructs a rotation matrix builder for the given dimension.
        :param dim:
        """
        # The dimension of rotation matrices produced by this builder.
        self.dim = dim
        # The 2D rotation planes currently used by this builder.
        self.planes = []

    def rotate_plane(self, i, j, theta):
        """
         Adds a rotation around the 2D rotation plane defined by the two axes.
          The plane is initially unrotated, but can be assigned a specific rotation
          angle if followed by setting theta.
        :param i:
        :param j:
        :return:
        """
        assert (i >= 0) and (i < self.dim) and (j >= 0) and (j < self.dim) and (i != j), 'invalid plane'
        self.planes.append(Plane(i, j, theta))

    def rotate_all_random(self):
        #rotation_variables = tf.get_variable('rotations', [self.dim, self.dim], dtype=tf.float32, initializer=tf.initializers.random_uniform(0, 1), constraint=lambda x: tf.maximum(tf.minimum(1.0, x), -1.0))
        rotation_variables = tf.get_variable('rotations', [self.dim, self.dim], dtype=tf.float32, initializer=tf.initializers.random_uniform(0, 1), constraint=lambda x: tf.maximum(tf.minimum(1.0, x), -1.0))
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                #plane = Plane(i, j, tf.Variable(float_uniform(0, 2 * math.pi), name='plane_' + str(i) + '_' + str(j)))
                plane = Plane(i, j, rotation_variables[i, j])
                self.planes.append(plane)

    def create(self):
        rotation = tf.eye(self.dim)
        #rotation = tf.SparseTensor(indices=[[i, i] for i in range(self.dim)], values=[1.0 for _ in range(self.dim)], dense_shape=[self.dim, self.dim])

        for plane in self.planes:
            old_rotation = rotation
            new_rotation = self.rotation_matrix(plane)
            #rotation = tf.matmul(old_rotation, tf.sparse_add(tf.zeros([self.dim, self.dim]), new_rotation))
            rotation = tf.sparse_tensor_dense_matmul(new_rotation, old_rotation)

        return rotation

    def rotation_matrix(self, plane):
        i = plane.first_axis
        j = plane.second_axis
        theta = plane.theta

        cos_theta = theta
        sin_thesta = tf.stop_gradient(tf.sin(tf.acos(theta)))
        # rotation = np.eye(self.dim).tolist()
        # rotation[i][i] = cos_theta
        # rotation[i][j] = -sin_thesta
        # rotation[j][i] = sin_thesta
        # rotation[j][j] = cos_theta
        # rotation = tf.stack(rotation)

        idx = []
        values = []
        for k in range(self.dim):
            idx.append([k, k])
            if k == i or k == j:
                values.append(cos_theta)
            else:
                values.append(1)
        idx.append([i, j])
        values.append(-sin_thesta)
        idx.append([j, i])
        values.append(sin_thesta)

        rotation = tf.SparseTensor(idx, values=tf.stack(values), dense_shape=[self.dim, self.dim])

        return rotation

class RotationMatrixBuilderTfGramSchmidt(object):
    def __init__(self, dim):
        self.dim = dim

    def create(self):
        vectors = tf.get_variable('unorthogonal_rotation', [self.dim, self.dim], dtype=tf.float32)
        # add batch dimension for matmul
        basis = tf.expand_dims(vectors[0,:]/tf.norm(vectors[0,:]),0)
        for i in range(1,vectors.get_shape()[0].value):
            v = vectors[i,:]
            # add batch dimension for matmul
            v = tf.expand_dims(v,0)
            w = v - tf.matmul(tf.matmul(v, basis, transpose_b=True), basis)
             # I assume that my matrix is close to orthogonal
            basis = tf.concat([basis, w/tf.norm(w)],axis=0)
        return basis