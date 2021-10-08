# Reliable Predictive Inference for Multivariate Response

An important factor to guarantee a responsible use of data-driven systems is that we should be able to communicate their uncertainty to decision makers. This can be accomplished by constructing predictive regions, which provide an intuitive measure of the limits of predictive performance.

This package contains a Python implementation of Spherically Transformed DQR (ST DQR) [1] methodology for constructing distribution-free predictive regions. 

# Calibrated Multiple-Output Quantile Regression with Representation Learning [1]

ST DQR is a method that reliably reports the uncertainty of a multivariate response and provably attains the user-specified coverage level.

[1] Shai Feldman, Stephen Bates, Yaniv Romano, [“Calibrated Multiple-Output Quantile Regression with Representation Learning.”](https://arxiv.org/abs/2110.00816) 2021.

## Getting Started

This package is self-contained and implemented in python.

Part of the code is a taken from the calibrated-quantile-uq package available at https://github.com/YoungseogChung/calibrated-quantile-uq. 
Part of the code is a taken from the oqr package available at https://github.com/Shai128/oqr. 


### Prerequisites

* python
* numpy
* scipy
* scikit-learn
* pytorch
* pandas

### Installing

The development version is available here on github:
```bash
git clone https://github.com/shai128/mqr.git
```

## Usage


### ST DQR

Please refer to [synthetic_example.ipynb](synthetic_example.ipynb) for basic usage. 
Comparisons to competitive methods and can be found in [display results.ipynb](display results.ipynb).

## Reproducible Research

The code available under /reproducible_experiments/ in the repository replicates the experimental results in [1].

### Publicly Available Datasets


* [Bio](https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure): Physicochemical  properties  of  protein  tertiary  structure  data  set.

* [House](https://www.kaggle.com/c/house-prices-advanced-regression-techniques): House prices.

* [Blog](https://archive.ics.uci.edu/ml/datasets/BlogFeedback): BlogFeedback data set.


### Data subject to copyright/usage rules

The Medical Expenditure Panel Survey (MPES) data can be downloaded using the code in the folder /get_meps_data/ under this repository. It is based on [this explanation](/get_meps_data/README.md) (code provided by [IBM's AIF360](https://github.com/IBM/AIF360)).

* [MEPS_19](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181): Medical expenditure panel survey,  panel 19.

* [MEPS_20](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181): Medical expenditure panel survey,  panel 20.

* [MEPS_21](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-192): Medical expenditure panel survey,  panel 21.



