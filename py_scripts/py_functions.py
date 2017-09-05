import numpy as np
import itertools
from collections import OrderedDict

## Construct Z to mimic FM - generates Z that has two way interactions.
#Used for initialization of Z
def fm_z(U):
    arr = np.arange(U)
    combs1 = list(itertools.combinations(arr, 1))
    combs2 = list(itertools.combinations(arr, 2))
    J1 = len(combs1)
    J2 = len(combs2)
    Z1 = np.zeros((U, J1), dtype=int)
    Z2 = np.zeros((U, J2), dtype=int)
    for j in range(J1):
            Z1[combs1[j],j] = 1
    for j in range(J2):
            Z2[combs2[j],j] = 1
    return Z1, Z2, J1, J2

# Controls train set predictions and gives error and bias for printing
def mean_error(y, p, w, p_l):
    p_n = p.sum(axis=(1,2))
    p_ln = p_l.sum(axis=1)
    predict = p_n + p_ln + w
    N = len(p_n)
    err = (predict - y)**2
    rmse = np.sqrt(err.sum()/N)
    return rmse
    
def mape_bias(y, p, w, p_l):
    p_n = p.sum(axis=(1,2))
    p_ln = p_l.sum(axis=1)
    predict = p_n + p_ln + w
    predict[predict<0] = 0.
    err = predict - y
    abs_e = np.absolute(err)
    return abs_e.sum()/y.sum(), err.sum()/y.sum()

# Initializing Gibbs sampler. 
def init(init_param):
    D, K, U, mu_0, gamma_0, J  = init_param
    Z1, Z2, J1, J2 = fm_z(U)
    Z = np.random.binomial(1, 0.3, (U,J))
    if J > (J1 + J2):
        print 'Too many interactions'
        Z[:,:J1] = Z1
        Z[:,J1:(J1+J2)] = Z2
    elif J < J1:
        print 'Too few interactions'
        ind = np.random.choice(J1, J, replace=False)
        Z = Z1[:,ind]
    else:
        Z[:,:J1] = Z1
        ind2 = np.random.choice(J2, J-J1, replace=False)
        Z[:,J1:] = Z2[:,ind2]
    V = np.random.normal(0.1, 0.1, (D,K))
    V = np.vstack((V,np.zeros((1,K))))
    w_l = np.append(np.random.normal(0.1, 0.1, D), 0)
    w = np.random.normal(0.1, 0.1)   
    mu_t = np.random.normal(mu_0, gamma_0)
    mu_v = np.random.normal(mu_0, gamma_0, K)
    lambda_v = np.random.gamma(1, gamma_0, K)
    return w, V, Z, w_l, mu_t, mu_v, lambda_v, J

    
# Samplers here
# Computes p_l, used after initialization
def get_p_l(p_l_param, w_l):
    N, U, cat, raw, X_c, X_r, raw_i = p_l_param
    p_l = np.zeros((N,U))
    p_l[:,cat] = w_l[X_c]
    p_l[:,raw] = X_r * w_l[raw_i]
    #p_l[:,raw] = 0 to avoid raw
    return p_l


# Resampling prior scales. Fast. Reduces sensitivity to prior parameters choice
def sample_scales(scale_param, w, w_l, V, p, p_l, mu_t, mu_v, lambda_v):
    alpha_0, beta_0, gamma_0, mu_0, alpha_l, beta_l, D, N, K, y, lin_model = scale_param
    p_n = p.sum(axis=(1,2))
    if lin_model == 0:
        p_l = np.zeros((N,D), dtype=float)
    y_n = p_n + p_l.sum(axis=1) + w
    #Update alpha
    alpha_n = (alpha_0 + N)/2 #
    beta_n = (((y-y_n)**2).sum() + beta_0)/2
    alpha = np.random.gamma(alpha_n, 1/beta_n)   
    #Updating prior for linear coefs and bias
    if lin_model == 0:
        alpha_t_l = (alpha_l + 1)/2 #
        beta_t_l = ((w-mu_t)**2 + beta_l)/2
        lambda_t = np.random.gamma(alpha_t_l, 1/beta_t_l)
        lambda_mu_t = (1 + gamma_0)*lambda_t 
        mu_mu_t = lambda_t*(gamma_0*mu_0 + w)/lambda_mu_t
        mu_t = np.random.normal(mu_mu_t, np.sqrt(1/lambda_mu_t)) 
    else:
        alpha_t_l = (alpha_l + D + 1)/2 #
        beta_t_l = (((w_l-mu_t)**2).sum() + (w-mu_t)**2 + beta_l)/2
        lambda_t = np.random.gamma(alpha_t_l, 1/beta_t_l)
        lambda_mu_t = (D + 1 + gamma_0)*lambda_t 
        mu_mu_t = lambda_t*(gamma_0*mu_0 + (w + w_l.sum()))/lambda_mu_t
        mu_t = np.random.normal(mu_mu_t, np.sqrt(1/lambda_mu_t))   
    #Updating prior for V
    for k in range(K):
        alpha_t_v = (alpha_l + D)/2 #
        beta_t_v = (((V[:,k]-mu_v[k])**2).sum() + beta_l)/2
        lambda_v[k] = np.random.gamma(alpha_t_v, 1/beta_t_v)
        lambda_mu_v = (gamma_0 + D)*lambda_v[k] #
        mu_mu_v = lambda_v[k] *(gamma_0*mu_0 + V[:,k].sum())/lambda_mu_v
        mu_v[k] = np.random.normal(mu_mu_v, np.sqrt(1/lambda_mu_v))       
    return alpha, mu_t, lambda_t, lambda_v, mu_v
