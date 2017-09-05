import numpy as np
import py_functions as ff
import g_c_sampler as gz
import sys
import copy
from joblib import Parallel, delayed

## Computing P(M) table of probabilities
def m_n_table(D, alpha, gamma1, gamma2):
    s = [reduce(lambda c, x: c+ [c[-1]*(alpha*x+gamma2)], range(D), [1.])]
    for i in range(1,D+1):
        new_r = [s[i-1][0]*(alpha*(i-1) + gamma1)]
        for j in range(1,D+1-i):
            new_r += [s[i-1][j]*(alpha*(i-1)+(1-alpha)*j+gamma1) + new_r[j-1]*((1-alpha)*i+alpha*(j-1)+gamma2)]
        s += [new_r]
    return s

## Improving initialization by trying some short starts
def init_restart(init_param, p_l_param, pr_param, scale_param, V_param, Z_param, y, seed, restart_iter, verbose, use_mape=False):
    np.random.seed(seed)
    w, V, Z, w_l, mu_t, mu_v, lambda_v, J = ff.init(init_param)
    p_l = ff.get_p_l(p_l_param, w_l)
    p = gz.get_p(pr_param, V, Z)
    for r in range(restart_iter):
        priors = ff.sample_scales(scale_param, w, w_l, V, p, p_l, mu_t, mu_v, lambda_v)
        w, w_l, V, p_l, p = gz.sample_w_V(V_param, p, p_l, w_l, V, Z, priors)
        Z , p = gz.sampleZ(Z_param, p, p_l, Z, V, w, priors[0])
    if use_mape:
        rmse, _ = ff.mape_bias(y, p, w, p_l)
    else:
        rmse = ff.mean_error(y, p, w, p_l)
    if verbose:
        if use_mape:
             print 'Seed %d MAPE is %f%%' % (seed, 100*rmse)
        else:
             print 'Seed %d RMSE is %f' % (seed, rmse)
        sys.stdout.flush()
    return [w, copy.deepcopy(w_l), copy.deepcopy(V), copy.deepcopy(Z), copy.deepcopy(priors), rmse, seed]

## Training MiFM
def train_gibbs_gauss(X, y, cat_to_v, v_to_cat, K, it, J, lin_model=True, alpha=1., verbose=True, restart=5, restart_iter=50, thr=300, rate=25, ncores=1, use_mape=False): #cat_list is cat_to_v; cat_keys are ceys of v_to_cat

    # From data - N observations; D variables; U categories
    N, D = X.shape
    U = len(v_to_cat)

    # Create categories hierarchy
    H = []
    for k in v_to_cat.keys():
        h_k = np.array([i for i in range(len(cat_to_v)) if k==cat_to_v[i][0]])
        H.append(h_k)
    
    ## Indeces of raw and categorical variables
    raw_i = np.array([i for i in range(len(cat_to_v)) if cat_to_v[i][1]=='raw'])
    raw = np.array([i for i in range(len(H)) if all([x in raw_i for x in H[i]])], dtype=int)
    cat = np.array([i for i in range(len(H)) if all([x not in raw_i for x in H[i]])], dtype=int)
    
    ## "Zero variable" index
    zind = D
    
    ## Splitting data into raw and categorical
    X_r = X[:,[H[i][0] for i in raw]]
    X_c = np.zeros((N, len(cat)), dtype=int)
    for i in range(len(cat)):
        ind = np.nonzero(X[:,H[cat[i]]])
        l_ind = np.repeat(zind, N)
        l_ind[ind[0]] = H[cat[i]][ind[1]]
        if len(set(ind[0]))!=len(ind[0]): print 'warning!'
        X_c[:,i] = l_ind
    
    # Variables to categories
    vc = np.array([[i,[k for k in range(U) if i in H[k]][0]] for i in range(D)])
    
    # Which observations are active for variables
    delta_ind = [np.nonzero(X_c[:,[k for k in range(len(cat)) if i in H[cat[k]]]]==i)[0] for i in range(D)]
    for r in raw_i:
        delta_ind[r] = np.arange(N)
    
    delta_ind = [n.astype(int) for n in delta_ind]


    # Hyperprioir pyrameters
    gamma_0 = 1.
    mu_0 = 0.
    alpha_0 = 1.
    beta_0 = 1.
    alpha_l = 1.
    beta_l = 1.

    ## FMM_alpha parameters and P(M) probability table
    hp_gammas = [1., 1.] # gamma_1 and gamma_2
    alpha = float(alpha)
    prob_table = m_n_table(U, alpha, hp_gammas[0], hp_gammas[1])
    
    ### Setting parameter sets for functions
    init_param = [D, K, U, mu_0, gamma_0, J]
    V_param = [y, X_r, raw_i, delta_ind, vc, int(lin_model)]
    scale_param = [alpha_0, beta_0, gamma_0, mu_0, alpha_l, beta_l, D, N, K, y, int(lin_model)]
    Z_param = [y, X_c, X_r, cat, 'NA', raw, prob_table, raw_i, alpha, hp_gammas]
    pr_param = [X_c, X_r, raw_i, raw]
    p_l_param = [N, U, cat, raw, X_c, X_r, raw_i]
    
    ### Initializing
    if type(restart)==list:
        w, w_l, V, Z, priors, rmse, seed = restart
        alpha, mu_t, lambda_t, lambda_v, mu_v = priors
        if verbose:
            if use_mape:
                print 'Starting based on seed %d with mape %f%%' % (seed, 100*rmse)
            else:
                print 'Starting based on seed %d with rmse %f' % (seed, rmse)
    elif restart>0:
        seeds = np.random.choice(100000, restart, replace=False)
        starts = Parallel(n_jobs=ncores)(delayed(init_restart)
        (init_param, p_l_param, pr_param, scale_param, V_param, Z_param, y, seeds[i], restart_iter, verbose, use_mape=use_mape) for i in range(restart))
        best = np.argmin([x[5] for x in starts])
        w, w_l, V, Z, priors, rmse, seed = starts[best]
        alpha, mu_t, lambda_t, lambda_v, mu_v = priors
        if verbose:
            if use_mape:
                print 'Seed %d chosen with mape %f%%' % (seed, 100*rmse)
            else:
                print 'Seed %d chosen with rmse %f' % (seed, rmse)
        if it==0:
            return starts[best]
    else:
        w, V, Z, w_l, mu_t, mu_v, lambda_v, J = ff.init(init_param)
        
        
    p_l = ff.get_p_l(p_l_param, w_l)
    samples = []
    p = gz.get_p(pr_param, V, Z)
    
    ## Gibbs sampler.
    for i in range(it):
        priors = ff.sample_scales(scale_param, w, w_l, V, p, p_l, mu_t, mu_v, lambda_v)
        w, w_l, V, p_l, p = gz.sample_w_V(V_param, p, p_l, w_l, V, Z, priors)
        Z , p = gz.sampleZ(Z_param, p, p_l, Z, V, w, priors[0])
        if (i + 1) % rate == 0 and i>thr:
            if verbose:
                if use_mape:
                    rmse, _ = ff.mape_bias(y, p, w, p_l)
                    print 'Iteration %d RMSE is %f%%' % (i, 100*rmse)
                else:
                    rmse = ff.mean_error(y, p, w, p_l)
                    print 'Iteration %d RMSE is %f' % (i, rmse)
                sys.stdout.flush()
            samples.append([w, copy.deepcopy(w_l), copy.deepcopy(V), copy.deepcopy(Z)])
            
    return samples