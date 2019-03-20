import numpy as np
import random
import string


def float_gaussian(mu, sigma, size=None):
    """
    Create a random float in the lower and upper bounds (normal distribution)
    :param mu:
        the expected value
    :param sigma:
        the distribution width
    :param size: int or tuple of ints, optional
        if None, function returns a single value, otherwise random values defined by the size
    :return:
        * a single float, if size=None
        * a numpy array (float32), if size is not None
    """
    values = np.random.normal(loc=float(mu), scale=float(sigma), size=size)
    values = np.cast['float32'](values)
    if values.shape == ():
        return float(values)
    return values


def float_uniform(low, high, size=None):
    """
    Create a random float in the lower and upper bounds (uniform distribution).
    :param low: float
        lower boundary
    :param high: float
        upper boundary
    :param size: int or tuple of ints, optional
        if None, function returns a single value, otherwise random values defined by the size
    :return:
        * a single float, if size=None
        * a numpy array (float32), if size is not None
    """
    values = np.random.uniform(low=float(low), high=float(high), size=size)
    values = np.cast['float32'](values)
    if values.shape == ():
        return float(values)
    return values


def int_gaussian(mu, sigma, size=None):
    """
    Create a random integer in the lower and upper bounds (gaussian distribution).
    :param mu:
        the expected value
    :param sigma:
        the distribution width
    :param size: int or tuple of ints, optional
        if None, function returns a single value, otherwise random values defined by the size
    :return:
        * a single integer, if size=None
        * a numpy array (int64), if size is not None
    """
    values = np.random.normal(loc=float(mu), scale=float(sigma), size=size)
    values = np.cast['int64'](values)
    if values.shape == ():
        return int(values)
    return values


def int_uniform(low, high, size=None):
    """
    Create a random integer in the lower and upper bounds (uniform distribution).
    :param low: int
        lower boundary
    :param high: int
        upper boundary
    :param size: int or tuple of ints, optional
        if None, function returns a single value, otherwise random values defined by the size
    :return:
        * a single integer, if size=None
        * a numpy array (int64), if size is not None
    """
    values = np.random.uniform(low=float(low), high=float(high), size=size)
    values = np.cast['int64'](values)
    if values.shape == ():
        return int(values)
    return values


def bool_uniform():
    """
    Create a random boolean, bit faster than numpy Bernoulli distribution with p=0.5.
    :return: True, or False
    """
    return bool(random.getrandbits(1))


def bool_bernoulli(p=0.5, size=None):
    """
    Create random booleans, with a given probability.

    :param p: float, or list of floats
        probabilities for the binomial distribution
    :param size: int or tuple of ints, optional
        if None, function returns a single value, otherwise random values defined by the size
    :return:
        * a single bool, if size=None
        * a numpy array of True's and/or False's, if size is not None
    """
    values = np.random.binomial(n=1, p=p, size=size)
    values = np.cast['bool'](values)
    if values.shape == ():
        return bool(values)
    return values


def log_uniform(low, high):
    """
    Generates a number that's uniformly distributed in the log-space between
    `low` and `high`

    :param: low : float
        Lower bound of the randomly generated number
    :param: high : float
        Upper bound of the randomly generated number

    :returns: rval : float
        Random number uniformly distributed in the log-space specified by `low`
        and `high`
    """
    log_low = np.log(low)
    log_high = np.log(high)

    log_rval = np.random.uniform(log_low, log_high)
    rval = float(np.exp(log_rval))

    return rval


def id_generator(size=8, chars=string.ascii_lowercase + string.digits):
    """
    Generate a random identifier from a given set of characters.

    :param size: int
        size of the id
    :param chars: str
        pool of characters used for the identifier
    :return: str
        the identifier
    """
    n_chars = len(chars)
    idcs = np.random.choice(n_chars, size)
    return ''.join(chars[i] for i in idcs)
