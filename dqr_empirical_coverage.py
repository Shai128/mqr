from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import chi2
import matplotlib as mpl

from helper import create_folder_if_it_doesnt_exist


def get_actual_coverage(nominal_coverage, d):
    R = norm.ppf(nominal_coverage)
    cov = chi2.cdf(R ** 2, d)
    return cov


if __name__ == '__main__':
    create_folder_if_it_doesnt_exist("figures")
    alpha = 0.1
    mpl.rc('font', **{'size': 15})
    actual_coverages = []
    nominal_coverage = 1 - alpha
    y_dims = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for d in y_dims:
        actual_coverages += [get_actual_coverage(nominal_coverage, d) * 100]
    plt.plot(y_dims, actual_coverages, linewidth=7)
    plt.xlabel("The dimension of the response")
    plt.ylabel("Empirical coverage rate (%)")
    plt.savefig("figures/Probability to be inside the quantile region as a function of r.png", dpi=300, bbox_inches='tight')
    plt.show()

    nominal_coverage_levels = np.arange(0.5, 1, 0.0001)
    d = 3
    actual_coverages = get_actual_coverage(nominal_coverage_levels, d) * 100
    nominal_coverage_levels = nominal_coverage_levels * 100
    plt.plot(nominal_coverage_levels, actual_coverages, linewidth=7)
    plt.plot([50, 100], [0, 100], 'k--', linewidth=4)
    plt.xlabel("Nominal coverage level (%)")
    plt.ylabel("Empirical coverage rate (%)")
    # plt.legend()
    plt.savefig("figures/Empirical coverage as a function of the nominal level, for r=3.png", dpi=300, bbox_inches='tight')
    plt.show()
