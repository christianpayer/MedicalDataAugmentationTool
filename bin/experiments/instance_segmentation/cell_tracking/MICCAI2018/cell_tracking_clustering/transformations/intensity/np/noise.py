
import numpy as np

def gaussian(image, stddev):
    return image + np.random.normal(scale=stddev, size=image.shape)
