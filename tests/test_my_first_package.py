"""Unit tests for array, matrix, and statistical NumPy operations."""

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


class TestArrayOperations:
    """Tests for element-wise array operations."""

    def test_elementwise_multiplication(self):
        array_a = np.array([1, 2, 3])
        array_b = np.array([4, 5, 6])

        result = elementwise_multiplication(array_a, array_b)

        np.testing.assert_array_equal(result, np.array([4, 10, 18]))

    def test_multiplication_with_broadcasting(self):
        array_a = np.array([[1, 2], [3, 4]])
        array_b = np.array([10, 20])

        result = elementwise_multiplication(array_a, array_b)

        np.testing.assert_array_equal(result, np.array([[10, 40], [30, 80]]))

    def test_multiplication_with_zero_and_negative_values(self):
        result = elementwise_multiplication(
            np.array([-2, 0, 3]), np.array([4, -5, 0])
        )

        np.testing.assert_array_equal(result, np.array([-8, 0, 0]))

    def test_multiplication_with_incompatible_shapes(self):
        with pytest.raises(ValueError):
            elementwise_multiplication(np.ones((2, 2)), np.ones(3))

    def test_elementwise_division(self):
        result = elementwise_division(
            np.array([8, 10, 12]), np.array([2, 5, 3])
        )

        np.testing.assert_allclose(result, np.array([4.0, 2.0, 4.0]))

    def test_division_by_zero(self):
        with pytest.warns(RuntimeWarning):
            result = elementwise_division(
                np.array([1.0, 0.0, -1.0]), np.array([0.0, 0.0, 0.0])
            )

        assert np.isposinf(result[0])
        assert np.isnan(result[1])
        assert np.isneginf(result[2])

    def test_division_with_incompatible_shapes(self):
        with pytest.raises(ValueError):
            elementwise_division(np.ones((2, 2)), np.ones(3))


class TestMatrixOperations:
    """Tests for matrix addition and subtraction."""

    def test_matrix_addition(self):
        matrix_a = np.array([[1, 2], [3, 4]])
        matrix_b = np.array([[5, 6], [7, 8]])

        result = matrix_addition(matrix_a, matrix_b)

        np.testing.assert_array_equal(result, np.array([[6, 8], [10, 12]]))

    def test_matrix_subtraction(self):
        matrix_a = np.array([[5, 6], [7, 8]])
        matrix_b = np.array([[1, 2], [3, 4]])

        result = matrix_subtraction(matrix_a, matrix_b)

        np.testing.assert_array_equal(result, np.array([[4, 4], [4, 4]]))

    def test_matrix_operations_with_negative_values(self):
        matrix_a = np.array([[-1, 0], [2, -3]])
        matrix_b = np.array([[4, -5], [-2, 3]])

        np.testing.assert_array_equal(
            matrix_addition(matrix_a, matrix_b), np.array([[3, -5], [0, 0]])
        )
        np.testing.assert_array_equal(
            matrix_subtraction(matrix_a, matrix_b), np.array([[-5, 5], [4, -6]])
        )

    @pytest.mark.parametrize("operation", [matrix_addition, matrix_subtraction])
    def test_matrix_operations_with_incompatible_shapes(self, operation):
        with pytest.raises(ValueError):
            operation(np.ones((2, 2)), np.ones((2, 3)))

    def test_matrix_operations_with_empty_matrices(self):
        matrix_a = np.empty((0, 2))
        matrix_b = np.empty((0, 2))

        np.testing.assert_array_equal(matrix_addition(matrix_a, matrix_b), matrix_a)
        np.testing.assert_array_equal(
            matrix_subtraction(matrix_a, matrix_b), matrix_a
        )


class TestStatisticsOperations:
    """Tests for mean, median, and standard deviation."""

    def test_statistics_for_even_sized_array(self):
        values = np.array([1, 2, 3, 4])

        assert array_mean(values) == pytest.approx(2.5)
        assert array_median(values) == pytest.approx(2.5)
        assert array_standard_deviation(values) == pytest.approx(
            np.std(values)
        )

    def test_statistics_for_odd_sized_array(self):
        values = np.array([7, 1, 5])

        assert array_mean(values) == pytest.approx(13 / 3)
        assert array_median(values) == pytest.approx(5)

    def test_statistics_with_negative_and_float_values(self):
        values = np.array([-2.5, 0.0, 2.5])

        assert array_mean(values) == pytest.approx(0.0)
        assert array_median(values) == pytest.approx(0.0)
        assert array_standard_deviation(values) == pytest.approx(
            np.sqrt(25 / 6)
        )

    def test_statistics_for_single_value(self):
        values = np.array([42])

        assert array_mean(values) == pytest.approx(42)
        assert array_median(values) == pytest.approx(42)
        assert array_standard_deviation(values) == pytest.approx(0)

    def test_statistics_for_constant_values(self):
        values = np.full(5, 7)

        assert array_mean(values) == pytest.approx(7)
        assert array_median(values) == pytest.approx(7)
        assert array_standard_deviation(values) == pytest.approx(0)

    def test_statistics_for_empty_array(self):
        values = np.array([])

        with pytest.warns(RuntimeWarning):
            mean = array_mean(values)
        with pytest.warns(RuntimeWarning):
            median = array_median(values)
        with pytest.warns(RuntimeWarning):
            standard_deviation = array_standard_deviation(values)

        assert np.isnan(mean)
        assert np.isnan(median)
        assert np.isnan(standard_deviation)

    def test_statistics_ignore_no_values_only_if_explicitly_supported(self):
        """Document current behavior: NaN values propagate through NumPy."""
        values = np.array([1.0, np.nan, 3.0])

        assert np.isnan(array_mean(values))
        assert np.isnan(array_median(values))
        assert np.isnan(array_standard_deviation(values))
