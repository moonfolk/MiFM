import pandas as pd
import StringIO as io
import requests

import sys
import os

cur_dir =  os.path.dirname(os.path.realpath('abalone_example.py'))
path_gs = cur_dir + '/py_scripts'
path_cython = cur_dir + '/cython'
path_predict = cur_dir + '/prediction'
sys.path.insert(0, path_gs)
sys.path.insert(0, path_cython)
sys.path.insert(0, path_predict)


from mifm_class import get_rmse, MiFM, pd_proc, get_chosen, get_combs, add_inters
import numpy as np
from sklearn.model_selection import train_test_split

## Reading data
url="http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
s=requests.get(url).content
column_names = ["sex", "length", "diameter", "height", "whole weight", 
                "shucked weight", "viscera weight", "shell weight", "rings"]
data=pd.read_csv(io.StringIO(s.decode('utf-8')), names=column_names)

## Processing data (i.e. one-hot encoding) and separating response
y = data.rings.values.astype(np.float)
del data['rings']
column_names.remove('rings')
X, v_to_cat, cat_to_v = pd_proc(data, column_names)

## Train-test split
np.random.seed(1)
train_X, test_X, train_y, test_y = train_test_split(X, y)

### Fitting with alpha=1
model_1 = MiFM(K=6, J=15, it=1500, rate=50, thr=500, restart=20, restart_iter=25, verbose=False, lin_model=False, alpha=1.)
model_1.fit(train_X, train_y, cat_to_v, v_to_cat)
print 'Test RMSE with alpha=1 and without linear terms is %f' % get_rmse(test_y, model_1.predict(test_X))
## z_1 gives marginal probabilities of interactions
# Values above 1 mean that interaction was repeated on some of the samples; '' means empty interaction
z_1 = get_chosen(model_1.samples_, v_to_cat, add_linear=False)

### Fitting with alpha=0.7
model_07 = MiFM(K=6, J=15, it=1500, rate=50, thr=500, restart=20, restart_iter=25, verbose=False, lin_model=False, alpha=0.7)
model_07.fit(train_X, train_y, cat_to_v, v_to_cat)
print 'Test RMSE with alpha=0.7 and without linear terms is %f' % get_rmse(test_y, model_07.predict(test_X))
## z_07 gives marginal probabilities of interactions
# Values above 1 mean that interaction was repeated on some of the samples; '' means empty interaction
z_07 = get_chosen(model_07.samples_, v_to_cat, add_linear=False)

### Fitting OLS with selected interactions
from sklearn.linear_model import LinearRegression
# Select interactions with marginals above thr
z_chosen_07 = get_chosen(model_07.samples_, v_to_cat, add_linear = False, thr=0.499)
combs_07 = get_combs(z_chosen_07, cat_to_v)
# Construct data sets with chosen interactions as features
MiFM_train_X = add_inters(train_X, combs_07, with_x = False)
MiFM_test_X = add_inters(test_X, combs_07, with_x = False)
# Fit OLS
model_ols_MiFM = LinearRegression()
model_ols_MiFM.fit(MiFM_train_X, train_y)
print 'Test RMSE for OLS with selected interactions is %f' % get_rmse(test_y, model_ols_MiFM.predict(MiFM_test_X))

### Compare to MLP
from sklearn.neural_network import MLPRegressor
D = train_X.shape[1]
model_nn = MLPRegressor(hidden_layer_sizes=(2*D,D,D/2),solver="lbfgs") # results with adam are worse
model_nn.fit(train_X,train_y)
print 'MLP test RMSE is %f' % get_rmse(test_y, model_nn.predict(test_X))