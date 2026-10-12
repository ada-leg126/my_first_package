"""Statistical calculations for NumPy arrays."""

import numpy as np


def array_mean(array):
    """Calculate the arithmetic mean of a NumPy array.

    Parameters
    ----------
    array : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.float64
        The mean of the array elements.
    """
    return np.mean(array)


def array_median(array):
    """Calculate the median of a NumPy array.

    Parameters
    ----------
    array : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.float64
        The median of the array elements.
    """
    return np.median(array)


def array_standard_deviation(array):
    """Calculate the standard deviation of a NumPy array.

    Parameters
    ----------
    array : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.float64
        The standard deviation of the array elements.
    """
    return np.std(array)
