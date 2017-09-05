import numpy as np

from g_c_sampler import predict_y
    
def pr_samples_cython(X, v_to_cat, cat_to_v, samples):
    raw_i = np.array([i for i in range(len(cat_to_v)) if cat_to_v[i][1]=='raw'])
    N, D = X.shape
    # Create categories hierarchy
    H = []
    for k in v_to_cat.keys():
        h_k = np.array([i for i in range(len(cat_to_v)) if k==cat_to_v[i][0]])
        H.append(h_k)
    
    ## Indeces of raw and categorical variables
    raw = np.array([i for i in range(len(H)) if all([x in raw_i for x in H[i]])])
    cat = np.array([i for i in range(len(H)) if all([x not in raw_i for x in H[i]])])
    
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
        
    pr_param = [X_c, X_r, raw_i, raw]
    result = np.zeros(N, dtype=float)
    l = len(samples)
    for m in samples:
        w = m[0]
        w_l = m[1]
        V = m[2]
        Z = m[3]
        pr = predict_y(pr_param, w, w_l, V, Z)
        result += pr
    return result/l