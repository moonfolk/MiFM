import itertools

import numpy as np
import train_f as gs
from predict_f_all import pr_samples_cython
from sklearn.feature_extraction import DictVectorizer
from collections import OrderedDict
from sklearn.base import BaseEstimator, RegressorMixin

## Data processing
def pd_proc(data, column_names, binary=''):
    v_to_cat = OrderedDict()
    cat_to_v = []
    N = len(data)
    np_data = np.ndarray((N,0), dtype=float)
    for i in column_names:
        temp = data[i].values
        if temp.dtype == np.dtype(object):
            if set(temp)==set(binary):
                pos = np.unique(temp)[0]
                dum_temp = (temp==pos).astype(int)
                np_data = np.column_stack((np_data, dum_temp))
                cat_to_v.append([i, pos])
                v_to_cat[i] = [pos]
            else:
                dum_dict = [{x: 1} for x in temp]
                v = DictVectorizer(sparse=False)
                dum_temp = v.fit_transform(dum_dict)
                names = v.get_feature_names()
                v_to_cat[i] = names
                np_data = np.column_stack((np_data, dum_temp))
                for n in names:
                    cat_to_v.append([i, n])
        else:
            cat_to_v.append([i, 'raw'])
            v_to_cat[i] = ['raw']
            np_data = np.column_stack((np_data, temp))
    return np_data, v_to_cat, cat_to_v

## Comparison scores
def get_rmse(true, predicted): ## indeed it is rms
    mse = ((predicted - true)**2).sum()/len(true)
    return np.sqrt(mse)

def mape_bias(y, predict):
    predict[predict<0] = 0.
    err = predict - y
    abs_e = np.absolute(err)
    return abs_e.sum()/y.sum(), err.sum()/y.sum()

## MiFM class for fitting Gibbs sampler
class MiFM(BaseEstimator, RegressorMixin):  
    def __init__(self, K=5, J=50, it=700, lin_model=True, alpha=1., verbose=False, restart=5, restart_iter=50, thr=300, rate=25, ncores=1, use_mape=False):

        self.K = K
        self.J = J
        self.it = it
        self.lin_model = lin_model
        self.alpha = alpha
        self.verbose = verbose
        self.restart = restart
        self.restart_iter = restart_iter
        self.thr = thr
        self.rate = rate
        self.ncores = ncores
        self.use_mape = use_mape
        
    def fit(self, X, y, cat_to_v, v_to_cat):
        self.cat_to_v_ = cat_to_v
        self.v_to_cat_ = v_to_cat
        self.samples_ = gs.train_gibbs_gauss(X, y, self.cat_to_v_, self.v_to_cat_, 
                        self.K, self.it, self.J, self.lin_model, self.alpha,
                        self.verbose, self.restart, self.restart_iter, self.thr, self.rate, self.ncores, self.use_mape)
        return self
    def predict(self, X):
        return pr_samples_cython(X, self.v_to_cat_, self.cat_to_v_, self.samples_)
    def score(self, X, y):
        if self.use_mape:
            rmse, _ = mape_bias(y, self.predict(X))
        else:
            rmse = get_rmse(y, self.predict(X))
        return -rmse

## Aggregate MCMC samples to get marginals of interactions
def get_chosen(samples, v_to_cat, add_linear = True, thr=0., select=None):
    Z_impact = OrderedDict()
    l = len(samples)
    it = 0
    for m in samples:
        Z = m[3]
        J = Z.shape[1]
        for j in range(J):
            inter = np.nonzero(Z[:,j])[0]
            if add_linear and len(inter) == 1:
                continue
            inter_cat = ', '.join([str(v_to_cat.keys()[i]) for i in inter])
            if inter_cat in Z_impact:
                Z_impact[inter_cat] += 1./l
            else:
                Z_impact[inter_cat] = 1./l
        it += 1
    
    Z_sorted_count = sorted(Z_impact.items(), key = lambda t: -t[1])
    if select is not None:
            if select in Z_impact:
                return min(1,Z_impact[select])
            else:
                return 0
    if thr:
        z_estim = [i[0] for i in Z_sorted_count if i[1]>thr and i[0]!='']
        if add_linear:
            z_estim = v_to_cat.keys() + z_estim
        return z_estim
    else:
        return Z_sorted_count

## Functions to construct data matrix with selected interactions as features 
def get_combs(chosen, cat_to_v, len_thr=0):
    if '' in chosen: chosen.remove('')
    cat_to_v = np.array(cat_to_v)
    combs_all = []
    for p in chosen:
        cats = p.split(', ')
        if len(cats)>len_thr:
            ind = [np.where(cat_to_v[:,0]==x)[0] for x in cats]
            combs_all += list(itertools.product(*ind))
    return combs_all

def add_inters(X, combs, with_x = False):
    N, D = X.shape
    inters = np.ndarray((N,0), dtype=float)
    for i in combs:
        n_col = np.prod(X[:,i], axis=1)
        inters = np.column_stack((inters, n_col))
    if with_x:
        return np.column_stack((X, inters))
    else:
        return inters