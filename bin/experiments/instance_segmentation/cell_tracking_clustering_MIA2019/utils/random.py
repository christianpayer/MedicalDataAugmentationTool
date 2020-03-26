
import numpy as np


def float_gaussian(mu, sigma, size=None):
    """
    Create random floats in the lower and upper bounds - normal distribution.
    :param mu: Mean value.
    :param sigma: Sigma value.
    :param size: If None, function returns a single value, otherwise random values in size of shape.
    :return: The random values.
    """
    values = np.random.normal(loc=float(mu), scale=float(sigma), size=size)
    if isinstance(values, np.ndarray):
        return values.astype(np.float32)
    return float(values)


def float_uniform(low, high, size=None):
    """
    Create random floats in the lower and upper bounds - uniform distribution.
    :param low: Minimum value.
    :param high: Maximum value.
    :param size: If None, function returns a single value, otherwise random values in size of shape.
    :return: The random values.
    """
    values = np.random.uniform(low=float(low), high=float(high), size=size)
    if isinstance(values, np.ndarray):
        return values.astype(np.float32)
    return float(values)


def int_gaussian(mu, sigma, size=None):
    """
    Create random integers in the lower and upper bounds - normal distribution.
    :param mu: Mean value.
    :param sigma: Sigma value.
    :param size: If None, function returns a single value, otherwise random values in size of shape.
    :return: The random values.
    """
    values = np.random.normal(loc=float(mu), scale=float(sigma), size=size)
    if isinstance(values, np.ndarray):
        return values.astype(np.uint64)
    return int(values)


def int_uniform(low, high, size=None):
    """
    Create random integers in the lower and upper bounds (uniform distribution).
    :param low: Minimum value.
    :param high: Maximum value.
    :param size: If None, function returns a single value, otherwise random values in size of shape.
    :return: The random values.
    """
    values = np.random.uniform(low=float(low), high=float(high), size=size)
    if isinstance(values, np.ndarray):
        return values.astype(np.uint64)
    return int(values)


def bool_bernoulli(p=0.5, size=None):
    """
    Create random booleans with a given probability.
    :param p: Probabilities for the binomial distribution.
    :param size: If None, function returns a single value, otherwise random values in size of shape.
    :return: The random values.
    """
    values = np.random.binomial(n=1, p=p, size=size)
    if isinstance(values, np.ndarray):
        return values.astype(np.bool)
    return bool(values)
