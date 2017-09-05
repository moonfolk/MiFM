# Multi-way Interacting Regression via Factorization Machines

This is a Python 2 implementation of MiFM algorithm for interaction discovery and prediction (M. Yurochkin, X. Nguyen, N. Vasiloglou to appear in NIPS 2017). Code written by Mikhail Yurochkin.

## Overview

This is a demonstration of MiFM on Abalone data. 

First compile cython code in cython folder. On Ubuntu run:
```
cython g_c_sampler.pyx
python setup.py build_ext --inplace
```

It implemets Gibbs sampling updates and prediction function

prediction/predict_f_all.py Python wrapper for Cython code to aggregate MCMC samples for prediction

py_scripts/train_f.py Python wrapper for Cython code to run Gibbs sampling

py_scripts/py_functions.py Gibbs sampling for hyperpriors and initialization

mifm_class.py Implements MiFM class; data preprocessing; posterior analysis of interactions

abalone_example.py downloads Abalone dataset and shows how to use MiFM and extract interactions

Implementation is designed to be used in the interactive mode (e.g. Python IDE like Spyder).

## Usage guide

```
MiFM(K=5, J=50, it=700, lin_model=True, alpha=1., verbose=False, restart=5, restart_iter=50, thr=300, rate=25, ncores=1, use_mape=False)
```

Parameters:

K: rank of matrix of coefficients V

J: number of interactions (columns) in Z

it: number of Gibbs sampling iterations

lin_model: whether to include linear effects (w_1,...,w_D)

alpha: FFM_alpha parameter. Smaller values encourage deeper interactions

verbose: whether to print intermediate RMSE train scores

restart and restart_iter: how many initializations to try with restart_iter iterations each. Then best initialization based on training RMSE is used for fitting

ncores: how many cores to use for initialization with restarts

use_mape: whether to use AMAPE instead of RMSE to select best initialization

thr: number of MCMC iterations after which samples are collected (i.e. burn-in)

rate: each rate iteration is saved

Methods:
```
fit(X, y, cat_to_v, v_to_cat)
```

X: training data after one-hot encoding

y: response

cat_to_v: list of category to value after one-hot encoding (see example with Abalone data)

v_to_cat: dictionary of category to values before one-hot encoding (see example with Abalone data)

Returns list of MCMC samples. Each sample is a list [bias, linear coefficients, V, Z]

```
predict(self, X)
```
Note: can only be used on fitted object. 
Returns predicted values for testing data X using Monte Carlo estimator of the mean response.

```
score(self, X, y)
```
Makes predictions and computes RMSE or AMAPE
