"""Contract tests for NumPy-backed array, matrix, and statistical operations.

The package API is expected to preserve NumPy semantics:
- element-wise and matrix operations support NumPy broadcasting;
- incompatible broadcast shapes raise ``ValueError``;
- operations return new results and do not mutate their inputs;
- statistics use NumPy defaults, including population standard deviation
  (``ddof=0``), propagation of ``NaN``, and ``NaN`` with ``RuntimeWarning``
  for empty inputs.
"""

import numpy as np
import pytest

from my_first_package.linear_algebra.array_operations import (
    elementwise_division,
    elementwise_multiplication,
)
from my_first_package.linear_algebra.matrix_operations import (
    matrix_addition,
    matrix_subtraction,
)
from my_first_package.statistics.statistics import (
    array_mean,
    array_median,
    array_standard_deviation,
)


@pytest.mark.parametrize(
    ("operation", "left", "right", "expected"),
    [
        (
            elementwise_multiplication,
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([4, 10, 18]),
        ),
        (
            elementwise_multiplication,
            np.array([[-2, 0], [3, -4]]),
            np.array([10, 20]),
            np.array([[-20, 0], [30, -80]]),
        ),
        (
            elementwise_division,
            np.array([8, 10, 12]),
            np.array([2, 5, 3]),
            np.array([4.0, 2.0, 4.0]),
        ),
        (
            elementwise_division,
            np.array([[2.0, 4.0], [6.0, 8.0]]),
            np.array([2.0, 4.0]),
            np.array([[1.0, 1.0], [3.0, 2.0]]),
        ),
    ],
)
def test_elementwise_operations_match_expected_values(
    operation, left, right, expected
):
    """Element-wise operations produce the expected values and shape."""
    result = operation(left, right)

    np.testing.assert_allclose(result, expected)
    assert result.shape == expected.shape


@pytest.mark.parametrize(
    "operation",
    [elementwise_multiplication, elementwise_division],
)
def test_elementwise_operations_reject_incompatible_broadcast_shapes(operation):
    with pytest.raises(ValueError):
        operation(np.ones((2, 2)), np.ones(3))


@pytest.mark.parametrize(
    ("operation", "left", "right"),
    [
        (
            elementwise_multiplication,
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
        ),
        (
            elementwise_division,
            np.array([8.0, 10.0, 12.0]),
            np.array([2.0, 5.0, 3.0]),
        ),
    ],
)
def test_elementwise_operations_do_not_mutate_inputs(operation, left, right):
    left_before = left.copy()
    right_before = right.copy()

    operation(left, right)

    np.testing.assert_array_equal(left, left_before)
    np.testing.assert_array_equal(right, right_before)


def test_division_by_zero_matches_numpy_semantics():
    """Floating-point division by zero warns and yields infinities/NaN."""
    numerator = np.array([1.0, 0.0, -1.0])
    denominator = np.zeros(3)

    with pytest.warns(RuntimeWarning):
        result = elementwise_division(numerator, denominator)

    np.testing.assert_array_equal(
        np.isposinf(result), np.array([True, False, False])
    )
    np.testing.assert_array_equal(
        np.isnan(result), np.array([False, True, False])
    )
    np.testing.assert_array_equal(
        np.isneginf(result), np.array([False, False, True])
    )


@pytest.mark.parametrize(
    ("operation", "left", "right", "expected"),
    [
        (
            matrix_addition,
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
            np.array([[6, 8], [10, 12]]),
        ),
        (
            matrix_subtraction,
            np.array([[5, 6], [7, 8]]),
            np.array([[1, 2], [3, 4]]),
            np.array([[4, 4], [4, 4]]),
        ),
        (
            matrix_addition,
            np.array([[-1, 0], [2, -3]]),
            np.array([4, -5]),
            np.array([[3, -5], [6, -8]]),
        ),
        (
            matrix_subtraction,
            np.array([[-1, 0], [2, -3]]),
            np.array([4, -5]),
            np.array([[-5, 5], [-2, 2]]),
        ),
    ],
)
def test_matrix_operations_support_numpy_broadcasting(
    operation, left, right, expected
):
    result = operation(left, right)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("operation", [matrix_addition, matrix_subtraction])
def test_matrix_operations_reject_incompatible_broadcast_shapes(operation):
    with pytest.raises(ValueError):
        operation(np.ones((2, 2)), np.ones((2, 3)))


@pytest.mark.parametrize("operation", [matrix_addition, matrix_subtraction])
def test_matrix_operations_support_empty_matrices(operation):
    left = np.empty((0, 2))
    right = np.empty((0, 2))

    result = operation(left, right)

    np.testing.assert_array_equal(result, left)
    assert result.shape == (0, 2)


@pytest.mark.parametrize(
    ("operation", "left", "right"),
    [
        (
            matrix_addition,
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
        ),
        (
            matrix_subtraction,
            np.array([[5, 6], [7, 8]]),
            np.array([[1, 2], [3, 4]]),
        ),
    ],
)
def test_matrix_operations_do_not_mutate_inputs(operation, left, right):
    left_before = left.copy()
    right_before = right.copy()

    operation(left, right)

    np.testing.assert_array_equal(left, left_before)
    np.testing.assert_array_equal(right, right_before)


@pytest.mark.parametrize(
    "values",
    [
        np.array([1, 2, 3, 4]),
        np.array([7, 1, 5]),
        np.array([-2.5, 0.0, 2.5]),
        np.array([42]),
        np.full(5, 7),
    ],
)
def test_statistics_match_numpy_defaults(values):
    """Statistics use NumPy's mean, median, and population std deviation."""
    assert array_mean(values) == pytest.approx(np.mean(values))
    assert array_median(values) == pytest.approx(np.median(values))
    assert array_standard_deviation(values) == pytest.approx(np.std(values))


def test_standard_deviation_uses_population_default():
    """Guard against changing from NumPy's ``ddof=0`` to sample std deviation."""
    values = np.array([-2.5, 0.0, 2.5])

    assert array_standard_deviation(values) == pytest.approx(np.sqrt(25 / 6))
    assert array_standard_deviation(values) != pytest.approx(np.std(values, ddof=1))


@pytest.mark.parametrize(
    "statistic",
    [array_mean, array_median, array_standard_deviation],
)
def test_statistics_for_empty_arrays_warn_and_return_nan(statistic):
    with pytest.warns(RuntimeWarning):
        result = statistic(np.array([]))

    assert np.isnan(result)


@pytest.mark.parametrize(
    "statistic",
    [array_mean, array_median, array_standard_deviation],
)
def test_statistics_propagate_nan(statistic):
    result = statistic(np.array([1.0, np.nan, 3.0]))

    assert np.isnan(result)