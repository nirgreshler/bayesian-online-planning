from typing import Tuple, List

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm

MIN_VAR = 1e-3

BIN_INDEX = 0       # Index of the bins in returned distribution
VALUE_INDEX = 0     # Index of values in returned distribution

APPROXIMATE_MAX_DISTRIBUTION = True  # True if to calculate the distribution of max{x1, x2, .., xn) by approximating


class ScalarDistribution:

    """
    Probability distribution for a continuous real-number.
    The PDF / CDF is assumed to be a piecewise constant / linear function, respectively.
    """

    def __init__(self, bins_start_values: np.ndarray, cdf_values: np.ndarray):
        """
        ctr
        :param bins_start_values: start value of each bin in which we describe the probability distribution
        :param cdf_values: the CDF in these bins. The CDF values should be a non-decreasing function, starting from 0
        -value and ending with 1-value.
        """
        assert np.all(np.diff(bins_start_values) > 0.) and np.all(np.diff(cdf_values) >= 0.)
        assert np.isclose(cdf_values[0], 0., atol=1e-6) and np.isclose(cdf_values[-1], 1., atol=1e-6)

        self._bins_start_values = bins_start_values
        self._cdf_values = cdf_values
        self._bins_start_values.setflags(write=False)
        self._cdf_values.setflags(write=False)

        self._pdf_values = None

        self._expectation = None
        self._std = None

        self._cdf_interpolator = None
        self._inverse_cdf_interpolator = None
        self._pdf_interpolator = None

    @property
    def cdf(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the bins and the CDF in these bins
        """
        return self._bins_start_values, self._cdf_values

    @property
    def pdf(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the bins and the PDF in these bins.
        Note: The PDF value of bin i corresponds to the PDF value in the range of bins [i-1, i]
        """
        if self._pdf_values is None:
            self._pdf_values = np.concatenate((np.array([0.]), np.diff(self._cdf_values) / np.diff(self._bins_start_values)))
        return self._bins_start_values, self._pdf_values

    @property
    def expectation(self) -> float:
        """
        Return the expectation of the distribution
        """
        if self._expectation is None:
            bins_values, pdf_values = self.pdf
            bins_center_values = (bins_values[:-1] + bins_values[1:]) / 2.
            self._expectation = np.sum(bins_center_values * pdf_values[1:] * np.diff(bins_values))
        return self._expectation

    @property
    def std(self) -> float:
        """
        Return the STD of the distribution
        """
        if self._std is None:
            bins_values, pdf_values = self.pdf
            bins_center_values = (bins_values[:-1] + bins_values[1:]) / 2.
            var = np.sum(bins_center_values * bins_center_values * pdf_values[1:] * np.diff(bins_values)) - \
                  self.expectation * self.expectation
            var = max(var, MIN_VAR)
            self._std = np.sqrt(var)
        return self._std

    def interpolate_cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Interpolate the CDF values in the requested locations.
        The CDF is assumed to be a piecewise linear function, so the interpolation is linear.
        :param x: the requested locations
        :return: the interpolated CDF values in the requested locations
        """
        if self._cdf_interpolator is None:
            self._cdf_interpolator = interp1d(self._bins_start_values,
                                              self._cdf_values,
                                              kind='linear',
                                              bounds_error=False,
                                              fill_value=(0., 1.))

        return self._cdf_interpolator(x)

    def interpolate_inverse_cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Interpolate the inverse CDF values in the requested locations.
        The CDF is assumed to be a piecewise linear function, so the interpolation is linear.
        :param x: the requested locations
        :return: the interpolated inverse CDF values in the requested locations
        """
        if self._inverse_cdf_interpolator is None:
            self._inverse_cdf_interpolator = interp1d(self._cdf_values, self._bins_start_values, kind='linear')

        return self._inverse_cdf_interpolator(x)

    def interpolate_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Interpolate the PDF values in the requested locations.
        The PDF is assumed to be a piecewise constant function, so interpolation is done by taking the PDF value of the
        next bin. We use a piecewise constant pdf so that cdf is simple and linear.
        :param x: the requested locations
        :return: the interpolated PDF values in the requested locations
        """
        if self._pdf_interpolator is None:
            bins_values, pdf_values = self.pdf
            self._pdf_interpolator = interp1d(bins_values,
                                              pdf_values,
                                              kind='next',
                                              bounds_error=False,
                                              fill_value=(0., 0.))

        return self._pdf_interpolator(x)

    @classmethod
    def create_normal_distribution(cls,
                                   start_percentile: float,
                                   stop_percentile: float,
                                   num_bins: int
                                   ) -> 'ScalarDistribution':
        """
        Create a normal distribution
        :param start_percentile: the numeric CDF / PDF will be approximated from this percentile
        :param stop_percentile: the numeric CDF / PDF will be approximated until this percentile
        :param num_bins: the number of bins of the numeric CDF / PDF
        :return: a ScalarDistribution which represents a normal distribution
        """
        normal_dist_bins_values = np.linspace(start=norm.ppf(start_percentile),
                                              stop=norm.ppf(stop_percentile),
                                              num=num_bins)
        normal_dist_cdf_values = norm.cdf(normal_dist_bins_values)
        normal_dist_cdf_values = cls.normalize_cdf_values(normal_dist_cdf_values)

        return ScalarDistribution(bins_start_values=normal_dist_bins_values, cdf_values=normal_dist_cdf_values)

    @classmethod
    def normalize_cdf_values(cls, cdf_values: np.ndarray) -> np.ndarray:
        """
        Normalize CDF values such that they will range from 0 to 1.
        It is assumed that the CDF values are arranged in increasing order.
        :param cdf_values: an un-normalized CDF values
        :return: a normalized CDF values
        """
        # Linear transform of the CDF values which makes the first(last) value to be equal to 0(1), respectively
        normalization_bias = -cdf_values[0] / (cdf_values[-1] - cdf_values[0])
        normalization_scale = 1. / (cdf_values[-1] - cdf_values[0])
        normalized_cdf_values = normalization_scale * cdf_values + normalization_bias

        # Clip values to avoid numerical errors
        normalized_cdf_values = np.clip(normalized_cdf_values, a_min=0., a_max=1.)
        normalized_cdf_values[0] = 0.
        normalized_cdf_values[-1] = 1.

        return normalized_cdf_values

    @classmethod
    def linear_transform(cls, distribution: 'ScalarDistribution', bias: float, scale: float) -> 'ScalarDistribution':
        """
        Calculate the distribution of a linear transform of a random variable
        :param distribution: a distribution of a random variable
        :param bias: the bias of the linear transform
        :param scale: the scale of the linear transform
        :return: the distribution of a linear transform of the random variable
        """
        bins_values, cdf_values = distribution.cdf
        new_bins_values = bias + scale * bins_values
        return ScalarDistribution(bins_start_values=new_bins_values, cdf_values=cdf_values)

    def __float__(self) -> float:
        return self.expectation

    def __str__(self):
        return f"ScalarDistribution: (Expectation: {self.expectation}, STD: {self.std})"


class DistributionTransformationUtils:
    """
    Utilities for Bayesian UCT calculations
    """
    def __init__(self):
        # TODO make constants
        distribution_percentile_range = [0.001, 0.999]  # numeric CDF / PDF are being approximated in this percentile range
        distribution_num_bins = 50  # the number of bins of the numeric CDF / PDF
        fine_distribution_percentile_range = [0.001, 0.9999999]  # numeric CDF / PDF are being approximated in this percentile range
        fine_distribution_num_bins = 1000  # the number of bins of the numeric CDF / PDF

        # Create a normal distribution for caching purposes
        self._normal_distribution = ScalarDistribution.create_normal_distribution(start_percentile=distribution_percentile_range[0],
                                                                                  stop_percentile=distribution_percentile_range[1],
                                                                                  num_bins=distribution_num_bins)
        self._normal_distribution_fine = ScalarDistribution.create_normal_distribution(start_percentile=fine_distribution_percentile_range[0],
                                                                                       stop_percentile=fine_distribution_percentile_range[1],
                                                                                       num_bins=fine_distribution_num_bins)

    def calculate_max_distribution(self, distributions: List[ScalarDistribution]) -> ScalarDistribution:
        """
        Calculate the distribution of the maximum of random variables
        :param distributions: distributions of random variables
        :return: the distribution of the maximum of these random variables
        """

        # TODO make constants
        distribution_num_bins_in_max_distribution_computation = 50  # the number of bins being used in the calculation of
        # the distribution of max{x1, x2, .., xn)
        distribution_percentile_range = [0.001,
                                         0.999]  # numeric CDF / PDF are being approximated in this percentile range
        distribution_num_bins = 50  # the number of bins of the numeric CDF / PDF

        # Extract data from the distributions
        first_bins_values, last_bins_values, expectations, stds, cdf_interpolators = [], [], [], [], []
        for dist in distributions:
            first_bins_values.append(dist.cdf[0][0])
            last_bins_values.append(dist.cdf[0][-1])
            expectations.append(dist.expectation)
            stds.append(dist.std)
            cdf_interpolators.append(dist.interpolate_cdf)

       # Create array of bins for the calculation
        max_first_bin_value = max(first_bins_values)  # any bin lower than that will have CDF value smaller than Config().mcts.bayesian_uct.distribution_percentile_range[0]
        max_last_bin_value = max(last_bins_values)  # any bin higher than that will have CDF value larger than Config().mcts.bayesian_uct.distribution_percentile_range[1]
        all_bins_values = np.linspace(start=max_first_bin_value,
                                      stop=max_last_bin_value,
                                      num=distribution_num_bins_in_max_distribution_computation)

        # Calculate the CDF values for each distribution in the requested bins
        if APPROXIMATE_MAX_DISTRIBUTION:
            # In case of a Gaussian approximation, treat each input distribution as a Gaussian, so we can vectorize
            # the computation of the CDF values by a single call for normal CDF interpolator.
            expectations = np.array(expectations)
            stds = np.array(stds)
            normalized_bins_values = (all_bins_values[None, :] - expectations[:, None]) / stds[:, None]
            cdf_values_per_dist = self._normal_distribution.interpolate_cdf(normalized_bins_values)
        else:
            # In case of an exact computation,
            cdf_values_per_dist = np.array([cdf_interpolator(all_bins_values) for cdf_interpolator in cdf_interpolators])

        # Calculate the numeric product of the CDFs
        all_cdf_values = np.prod(cdf_values_per_dist, axis=0)
        all_cdf_values = ScalarDistribution.normalize_cdf_values(all_cdf_values)

        # Calculate the bins for the output distribution
        min_bin_value = all_bins_values[all_cdf_values.size - 1 - np.argmin(np.abs(all_cdf_values[::-1] - distribution_percentile_range[0]))]
        max_bin_value = all_bins_values[np.argmin(np.abs(all_cdf_values - distribution_percentile_range[1]))]
        bins_values = np.linspace(start=min_bin_value, stop=max_bin_value, num=distribution_num_bins)

        # Calculate the CDF values for the output distribution
        cdf_interpolator = interp1d(all_bins_values, all_cdf_values, bounds_error=False, fill_value=(0., 1.))
        cdf_values = cdf_interpolator(bins_values)
        cdf_values = ScalarDistribution.normalize_cdf_values(cdf_values)

        return ScalarDistribution(bins_start_values=bins_values, cdf_values=cdf_values)

    def calculate_approximate_percentile_for_gaussian(self, means: np.ndarray, stds: np.ndarray, percentile: float) -> np.ndarray:
        """
        Calculate the percentile value for each input Gaussian distribution.
        :param means: the means of the input Gaussian distributions
        :param stds: the stds of the input Gaussian distributions
        :param percentile: the desired percentile (between 0 and 1)
        :return: array of the percentile values
        """
        normal_bins_values, normal_cdf_values = self._normal_distribution_fine.cdf
        normal_percentile_idx = np.argmin(np.abs(normal_cdf_values - percentile))
        normal_percentile_value = normal_bins_values[normal_percentile_idx]
        return means + normal_percentile_value * stds

    def create_gaussian_distribution(self, mean: float, std: float) -> ScalarDistribution:
        """
        Create a Gaussian distribution with the given mean and std.
        This method exploits a cached normal distribution for a faster computation.
        :param mean: the mean
        :param std: the std
        :return: the Gaussian distribution
        """
        return ScalarDistribution.linear_transform(distribution=self._normal_distribution, bias=mean, scale=std)