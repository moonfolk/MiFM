
from __future__ import division
import numpy as np
import copy
cimport numpy as np
Df = np.float
ctypedef np.float_t Df_t
Di = np.int
ctypedef np.int_t Di_t
cimport cython
from libc.math cimport exp, log, sqrt
#from libc cimport bool

@cython.profile(False)
cdef inline double square(double x): return x * x

#@cython.profile(False)
#cdef inline double prob_1(int i, int m, double alpha):
#    cdef double p0 = i - m + alpha
#    return p0/(p0-1)
#
#@cython.profile(False)
#cdef inline double prob_0(int m, double beta):
#    cdef double p0 = m + beta
#    return p0/(p0+1)
@cython.profile(False)
cdef inline double prob_1(int i, int m, double alpha_0, double gamma1):
    cdef double p1 = alpha_0*m + (1-alpha_0)*(i-m) + gamma1
    return p1/(p1-1+2*alpha_0)

@cython.profile(False)
cdef inline double prob_0(int i, int m, double alpha_0, double gamma2):
    cdef double p0 = (1-alpha_0)*m + alpha_0*(i-m) + gamma2
    return p0/(p0+1-2*alpha_0)

#@cython.boundscheck(False) # turn off bounds-checking for entire function
#@cython.wraparound(False)
#@cython.cdivision(True)
#@cython.nonecheck(False)
#@cython.profile(False)
#cdef inline double direct_j(np.ndarray[Di_t] z, int j, double alpha, double beta):
#    z[j] = 0
#    p_0_1 = (sum(z[:j]) + beta)/(j-sum(z[:j]) + alpha)
#    p_0_1_prod = [prob_1(i, sum(z[:i]), alpha) if z[i] else prob_0(sum(z[:i]), beta) for i in range(j+1, len(z))]
#    p_0_1 *= np.prod(p_0_1_prod)
#    return 1./(p_0_1 + 1.)

#@cython.boundscheck(False) # turn off bounds-checking for entire function
#@cython.wraparound(False)
#@cython.cdivision(True)
#@cython.nonecheck(False)
#@cython.profile(False)
#def direct_j(np.ndarray[Di_t, ndim=1] z, int j, double alpha, double beta):
#    cdef int zj = int(z[j]==1)
#    p_0_1 = (sum(z[:j]) + beta)/(j-sum(z[:j]) + alpha)
#    p_0_1_prod = [prob_1(i, sum(z[:i])-zj, alpha) if z[i] else prob_0(sum(z[:i])-zj, beta) for i in range(j+1, len(z))]
#    p_0_1 *= np.prod(p_0_1_prod)
#    return 1./(p_0_1 + 1.)
def direct_j(np.ndarray[Di_t, ndim=1] z, int j, double alpha_0, double gamma1, double gamma2):
    cdef int zj = int(z[j]==1)
    p_0_1 = ((1-alpha_0)*sum(z[:j]) + alpha_0*(j-sum(z[:j])) + gamma2)/((1-alpha_0)*(j-sum(z[:j])) + alpha_0*sum(z[:j]) + gamma1)
    p_0_1_prod = [prob_1(i, sum(z[:i])-zj, alpha_0, gamma1) if z[i] 
        else prob_0(i, sum(z[:i])-zj, alpha_0, gamma2) for i in range(j+1, len(z))]
    p_0_1 *= np.prod(p_0_1_prod)
    return 1./(p_0_1 + 1.)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.profile(False)
cdef double p_last_1(list table, double alpha, double gamma1, int m, int n):
    cdef double ratio = table[m-1][n-m]/table[m][n-m]
    cdef double p = (alpha*(m-1)+(1-alpha)*(n-m)+gamma1)*ratio
    return p


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.profile(False)
def sampleZ(Z_param, np.ndarray[Df_t, ndim=3] p, np.ndarray[Df_t, ndim=2] p_l, np.ndarray[Di_t, ndim=2] Z, np.ndarray[Df_t, ndim=2] V, double w, double alpha):
    assert p.dtype == Df and p_l.dtype == Df and Z.dtype == Di and V.dtype == Df
    #cdef np.ndarray[Di_t, ndim=2] Z = Z.copy()
    #y, X_c, X_r, cat, J, U, gamma, raw, H, raw_i = Z_param - pois model
    #y, X_c, X_r, cat, gamma, raw, H, raw_i = Z_param - gauss model
    cdef np.ndarray[Df_t, ndim=1] y = Z_param[0]
    cdef np.ndarray[Di_t, ndim=2] X_c = Z_param[1]
    cdef np.ndarray[Df_t, ndim=2] X_r = Z_param[2]
    cdef np.ndarray[Di_t, ndim=1] cat = Z_param[3]
    cdef int cat_len = cat.shape[0]
    cdef int J = Z.shape[1]
    cdef int U = Z.shape[0]
    cdef int K = V.shape[1]
    cdef int N = X_r.shape[0]
#    cdef double gamma = Z_param[4]
    cdef np.ndarray[Di_t, ndim=1] raw = Z_param[5]
    cdef np.ndarray[Di_t, ndim=1] raw_i = Z_param[7]
    
    ###### EBP parameters
    cdef list all_param = Z_param[9]
    cdef double gamma1 = all_param[0]
    cdef double gamma2 = all_param[1]
    cdef list table = Z_param[6]
    cdef double alpha_0 = Z_param[8] # 1 for deeper interactions
    
    ### Keeping track of interaction impacts and related
    cdef np.ndarray[Df_t, ndim=1] p_n = p.sum(axis=(1,2))
    cdef np.ndarray[Df_t, ndim=3] p_xvi = np.zeros((U,N,K), dtype=Df)
    cdef np.ndarray[Df_t, ndim=2] lambda_0_nk = np.zeros((N,K), dtype=Df)
    cdef np.ndarray[Df_t, ndim=1] p1_n = np.zeros(N, dtype=Df)
    cdef np.ndarray[Df_t, ndim=1] p0_n = np.zeros(N, dtype=Df)
    cdef np.ndarray[Di_t, ndim=1] m = Z.sum(axis=0)
    cdef np.ndarray[Df_t, ndim=1] p_ln = p_l.sum(axis=1)
    
    # integers and stuff
    cdef int old, i, j, n, k, cat_c, u, raw_c, count, raw_ind, cat_ind, m_ij
    cdef double temp, cur_kj
    cdef double p1_z, p0_z, prior_part, ll_part, log_p_dif, prob, y_dif_2, y_dif_1, p_last_is_1
    cdef int l_raw = raw.shape[0] - 1
    cdef int l_cat = cat.shape[0] - 1
    cdef list perm_J
    cdef list seen = []
    
    perm_J = np.random.permutation(J).tolist()
    for j in perm_J:
        for count in range(U):
            if m[j]>0:
                p_last_is_1 = p_last_1(table, alpha_0, gamma1, m[j], U)
                p_last_is_1 = min(1,p_last_is_1)
            else:
                p_last_is_1 = 0.
            old = np.random.binomial(1, p_last_is_1)
            i = np.random.choice(np.where(Z[:,j]==old)[0])
            if i not in seen:
                seen.append(i)
                if i in raw:
                    raw_ind = raw.tolist().index(i)
                    for n in range(N):
                        for k in range(K):
                            p_xvi[i,n,k] = X_r[n, raw_ind] * V[raw_i[raw_ind], k]
                else:
                    cat_ind = cat.tolist().index(i)
                    for n in range(N):
                        for k in range(K):
                            p_xvi[i,n,k] = V[X_c[n,cat_ind],k]

            y_dif_2 = 0. # y^2_0 - y^2_1
            y_dif_1 = 0. # 2y(y_1 - y_0)
            # Compute z prior part            
            m_ij = m[j] - Z[i,j]
#            p1_z = (m_ij + gamma)/(U + gamma)
            p1_z = (alpha_0*m_ij + (1-alpha_0)*(U-1-m_ij) + gamma1)/(U-1+gamma1+gamma2)
            p0_z = 1. - p1_z
            if old == 0:              
                for n in range(N):
                    temp = 0.
                    for k in range(K):
                        if m[j]>0:
                            temp += p[n,j,k]*(p_xvi[i,n,k] - 1.)
                        else:
                            temp += p_xvi[i,n,k]
                    p0_n[n] = p_n[n]
                    p1_n[n] = p_n[n] + temp
                    y_dif_2 += square(p0_n[n]+w+p_ln[n]) - square(p1_n[n]+w+p_ln[n])
                    y_dif_1 += y[n]*temp
            else:
                if m[j] > 1:
                    Z[i,j] = 0                
                    for n in range(N):
                        temp = 0.
                        if p_xvi[i,n,0]!=0.: # should be !=
                            for k in range(K):
                                temp += p[n,j,k]*(1/p_xvi[i,n,k] - 1.)
                        else:
                            for k in range(K):
                                cur_kj = 1.
                                cat_c = 0
                                raw_c = 0
                                for u in range(U):
                                    if Z[u,j]==1:
                                        if u==raw[raw_c]:
                                            cur_kj *= X_r[n,raw_c]*V[raw_i[raw_c],k]
                                        else:
                                            cur_kj *= V[X_c[n,cat_c],k]
                                    if u!=raw[raw_c]:
                                        if l_cat>cat_c:
                                            cat_c += 1
                                    elif l_raw>raw_c:
                                        raw_c += 1
                                lambda_0_nk[n,k] = cur_kj
                                temp += cur_kj
                        p1_n[n] = p_n[n]
                        p0_n[n] = p_n[n] + temp
                        y_dif_2 += square(p0_n[n]+w+p_ln[n]) - square(p1_n[n]+w+p_ln[n])
                        y_dif_1 -= y[n]*temp
                else:
                    for n in range(N):
                        p1_n[n] = p_n[n]
                        temp = 0.
                        for k in range(K):
                            temp += p[n,j,k]
                        p0_n[n] = p_n[n] - temp                      
                        y_dif_2 += square(p0_n[n]+w+p_ln[n]) - square(p1_n[n]+w+p_ln[n])
                        y_dif_1 += y[n]*temp
            ll_part = (-alpha/2)*(y_dif_2 + 2*y_dif_1)
            prior_part = log(p0_z) - log(p1_z)
            log_p_dif = ll_part + prior_part
            # Computing probability of 1            
            prob = 1/(1 + exp(log_p_dif))
#            if np.isnan(prob) or np.isinf(prob):
#                print prior_part
#                print ll_part
#                print i
#                print m_ij
#                print m[j]
#                print p0_z
#                print p1_z
#                print alpha, U, gamma1, gamma2
            # Sampling new value
            Z[i,j] = np.random.binomial(1, prob)
            # Updating affected parts of p if new value differs from old
            if Z[i,j] > old: # was 0, became 1
                for n in range(N):
                    for k in range(K):            
                        if m[j]>0:
                            p[n,j,k] *= p_xvi[i,n,k]
                        else:
                            p[n,j,k] += p_xvi[i,n,k]
                    p_n[n] = p1_n[n]
                m[j] += 1
            if Z[i,j] < old: # was 1, became 0
                if m[j]>1:
                    for n in range(N):
                        if p_xvi[i,n,0]>0. or p_xvi[i,n,0]<0.: # should be !=0
                            for k in range(K):
                                p[n,j,k] /= p_xvi[i,n,k]
                        else:
                            for k in range(K):
                                p[n,j,k] = lambda_0_nk[n,k]                          
                        p_n[n] = p0_n[n]
                else:
                    for n in range(N):
                        for k in range(K):
                            p[n,j,k] = 0.
                        p_n[n] = p0_n[n]
                m[j] -= 1
    return Z, p

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.profile(False)
def sample_w_V(V_param, np.ndarray[Df_t, ndim=3] p, np.ndarray[Df_t, ndim=2] p_l, np.ndarray[Df_t, ndim=1] w_l, np.ndarray[Df_t, ndim=2] V, np.ndarray[Di_t, ndim=2] Z, priors):
    assert p.dtype == Df and p_l.dtype == Df and Z.dtype == Di and V.dtype == Df and w_l.dtype == Df
    
    
    cdef double alpha = priors[0]
    cdef double mu_t = priors[1]
    cdef double lambda_t = priors[2]
    cdef np.ndarray[Df_t, ndim=1] lambda_v = priors[3]
    cdef np.ndarray[Df_t, ndim=1] mu_v = priors[4]
    
    #y, X_r, raw_i, delta_ind, vc = V_param
    cdef np.ndarray[Df_t, ndim=1] y = V_param[0]
    cdef np.ndarray[Df_t, ndim=2] X_r = V_param[1]
    cdef int D = V.shape[0] - 1
    cdef int K = V.shape[1]
    cdef int J = Z.shape[1]
    cdef int U = Z.shape[0]
    cdef int N = X_r.shape[0]
    cdef np.ndarray[Di_t, ndim=1] raw_i = V_param[2]
    cdef list delta_ind = V_param[3]
    cdef np.ndarray[Di_t, ndim=1] delta_value
    cdef np.ndarray[Di_t, ndim=2] vc = V_param[4]
    cdef int lin_model = V_param[5]
    #cdef list J_of_i
    
    cdef np.ndarray[Df_t, ndim=1] p_n = p.sum(axis=(1,2))
    
    cdef int delta_len
    cdef int u_of_i
    cdef double h_v_n, g_v_n, mu_all, sigma_all, sigma_v, mu_star_v
    cdef int i, j, k, d_cur, k1, u
    #cdef int j_count
    cdef int d
    cdef double w
    cdef int raw_c = 0
    cdef int l_raw = raw_i.shape[0] - 1
    
    ## Bias and linear variables
    cdef double s_w, mu_w, mu_l, s_l
    #cdef np.ndarray[Df_t, ndim=1] s_l = np.zeros(D, dtype=Df)
    cdef np.ndarray[Df_t, ndim=1] p_ln = p_l.sum(axis=1)
    cdef list perm_D
    cdef np.ndarray[Di_t, ndim=1] raw_order
    
    if lin_model == 0:
        p_ln = np.zeros(N, dtype=Df)
    #Updating bias
    s_w = 1/(lambda_t + alpha*N)
    mu_w = 0.
    for n in range(N):
        mu_w += y[n] - p_n[n] - p_ln[n]
    #mu_w = s_w*(alpha*(y-p_n-p_ln).sum()+mu_t*lambda_t)
    mu_w = s_w*(alpha*mu_w + mu_t*lambda_t)
    w = np.random.normal(mu_w, sqrt(s_w))
    
    #Updating linear coefs
#    h_lin = np.array([len(x) for x in delta_ind], dtype=float) #
#    h_lin[raw_i] = (X_r**2).sum() #
#    s_l = 1/(lambda_t + alpha*h_lin)
#    for i in range(D):
#        if i==raw_i:
#            s_l[i] = 1/(lambda_t + alpha*)
    if lin_model == 0:
        w_l = np.zeros(D+1, dtype=Df)
        p_l = np.zeros((N,D), dtype=Df)
    else:
        perm_D = np.random.permutation(D).tolist()
        raw_order = np.argsort([perm_D.index(x) for x in raw_i])
        for i in perm_D:
            mu_l = 0.
            u_of_i = vc[i,1]
            if i != raw_i[raw_order[raw_c]]:
                delta_value = delta_ind[i]
                delta_len = delta_value.shape[0]
                for d_cur in range(delta_len):
                    d = delta_value[d_cur]
                    p_ln[d] -= w_l[i]
                    mu_l += y[d] - w - p_n[d] - p_ln[d]
                s_l = 1/(lambda_t + alpha*delta_len)
                mu_l = s_l*(alpha*mu_l + mu_t*lambda_t)
                w_l[i] = np.random.normal(mu_l, sqrt(s_l))
                for d_cur in range(delta_len):
                    d = delta_value[d_cur]
                    p_l[d, u_of_i] = w_l[i]
                    p_ln[d] += w_l[i]
            else:
                s_l = 0
                for n in range(N):
                    p_ln[n] -= w_l[i]*X_r[n,raw_order[raw_c]]
                    mu_l += (y[n] - w - p_n[n] - p_ln[n])*X_r[n,raw_order[raw_c]]       
                    s_l += square(X_r[n,raw_order[raw_c]])
                s_l = 1/(lambda_t + alpha*s_l)
                mu_l = s_l*(alpha*mu_l + mu_t*lambda_t)
                w_l[i] = np.random.normal(mu_l, sqrt(s_l))
                for n in range(N):
                    p_l[n, u_of_i] = w_l[i] * X_r[n,raw_order[raw_c]]
                    p_ln[n] += w_l[i]*X_r[n,raw_order[raw_c]]
                if l_raw>raw_c:
                    raw_c += 1
                
                
#            p_l[delta_ind[i], vc[i,1]] = 0
#            g_lin = p_l[delta_ind[i],:].sum(axis=1) + p_n[delta_ind[i]] + w
#            mu_l = s_l[i]*(alpha*(y[delta_ind[i]] - g_lin).sum() + mu_t*lambda_t)
#            w_l[i] = np.random.normal(mu_l, sqrt(s_l[i]))
#            #print mu_l
#            p_l[delta_ind[i], vc[i,1]] = w_l[i]
#        else:
#            p_l[:, vc[i,1]] = 0
#            g_lin = p_l.sum(axis=1) + p_n + w
#            mu_l = s_l[i]*(alpha*((y - g_lin)*X_r).sum() + mu_t*lambda_t)
#            w_l[i] = np.random.normal(mu_l, sqrt(s_l[i]))
#            p_l[:, vc[i,1]] = w_l[i] * X_r
    
    #Updating interaction coeffs
    #for i in range(V.shape[0]-1):
    #p_ln = p_l.sum(axis=1)
    perm_D = np.random.permutation(D).tolist()
    for i in perm_D:
        delta_value = delta_ind[i]
        delta_len = delta_value.shape[0]
        u_of_i = vc[i,1]
        perm_K = np.random.permutation(K).tolist()
        for k in perm_K:      
            mu_all = 0.
            sigma_all = 0.
            for d_cur in range(delta_len):
                d = delta_value[d_cur]
                h_v_n = 0.
                g_v_n = w + p_ln[d]
                for k1 in range(K):
                    if k1!=k:
                        for j in range(J):
                            g_v_n += p[d,j,k1]                        
                #j_count = 0
                for j in range(J):
                    #if j in J_of_i:
                    if Z[u_of_i, j] == 1:                   
                        p[d,j,k] /= V[i,k]
                        h_v_n += p[d,j,k]
                        #j_count += 1
                    else:
                        g_v_n += p[d,j,k]
                mu_all += h_v_n*(y[d] - g_v_n)
                sigma_all += square(h_v_n)
            sigma_v = 1/(alpha*sigma_all + lambda_v[k])
            mu_star_v = sigma_v*(alpha*mu_all + mu_v[k]*lambda_v[k])
            V[i,k] = np.random.normal(mu_star_v, sqrt(sigma_v))
            for d_cur in range(delta_len):
                d = delta_value[d_cur]
                #j_count = 0
                for j in range(J):
                    #if j in J_of_i:
                    if Z[u_of_i, j] == 1:
                        p[d,j,k] *= V[i,k]
                        #j_count += 1
                
            
                
    return w, w_l, V, p_l, p

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.profile(False)
def get_p(data, np.ndarray[Df_t, ndim=2] V, np.ndarray[Di_t, ndim=2] Z):
    #assert p.dtype == Df and p_l.dtype == Df and Z.dtype == Di and V.dtype == Df
    #cdef np.ndarray[Di_t, ndim=2] Z = Z.copy()
    #y, X_c, X_r, cat, J, U, gamma, raw, H, raw_i = Z_param - pois model
    #y, X_c, X_r, cat, gamma, raw, H, raw_i = Z_param - gauss model
    cdef np.ndarray[Di_t, ndim=2] X_c = data[0]
    cdef np.ndarray[Df_t, ndim=2] X_r = data[1]
    #cdef np.ndarray[Di_t, ndim=1] cat = np.array(data[2])
    cdef np.ndarray[Di_t, ndim=1] raw_i = data[2]
    cdef np.ndarray[Di_t, ndim=1] raw = data[3]
    cdef int l_raw = raw.shape[0] - 1
    cdef int l_cat = X_c.shape[1] - 1
    
    cdef int J = Z.shape[1]
    cdef int U = Z.shape[0]
    cdef int K = V.shape[1]
    cdef int N = X_r.shape[0]
    #cdef int D = V.shape[0] - 1
    
    cdef np.ndarray[Df_t, ndim=3] p = np.zeros((N,J,K), dtype=Df)
    
    cdef double cur_kj
    
    cdef int n, j, u, k, cat_c, raw_c
    
    cdef bint flag
    
    
    for n in range(N):
        for j in range(J):
            for k in range(K):
                cur_kj = 1.
                cat_c = 0
                raw_c = 0
                flag = False
                for u in range(U):
                    if Z[u,j]==1:
                        flag = True
                        if u==raw[raw_c]:
                            cur_kj *= X_r[n,raw_c]*V[raw_i[raw_c],k]
                        else:
                            cur_kj *= V[X_c[n,cat_c],k]
                    if u!=raw[raw_c]:
                        if l_cat>cat_c:
                            cat_c += 1
                    elif l_raw>raw_c: 
                        raw_c += 1
                if flag:
                    p[n,j,k] = cur_kj
                else:
                    p[n,j,k] = 0.
    return p
    
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.profile(False)
def predict_y(data, double w, np.ndarray[Df_t, ndim=1] w_l, np.ndarray[Df_t, ndim=2] V, np.ndarray[Di_t, ndim=2] Z):
    assert w_l.dtype == Df and V.dtype == Df and Z.dtype == Di
    #cdef np.ndarray[Di_t, ndim=2] Z = Z.copy()
    #y, X_c, X_r, cat, J, U, gamma, raw, H, raw_i = Z_param - pois model
    #y, X_c, X_r, cat, gamma, raw, H, raw_i = Z_param - gauss model
    cdef np.ndarray[Di_t, ndim=2] X_c = data[0]
    cdef np.ndarray[Df_t, ndim=2] X_r = data[1]
    #cdef np.ndarray[Di_t, ndim=1] cat = np.array(data[2])
    cdef np.ndarray[Di_t, ndim=1] raw_i = data[2]
    cdef np.ndarray[Di_t, ndim=1] raw = data[3]
    cdef int l_raw = raw.shape[0] - 1
    cdef int l_cat = X_c.shape[1] - 1
    
    cdef int J = Z.shape[1]
    cdef int U = Z.shape[0]
    cdef int K = V.shape[1]
    cdef int N = X_r.shape[0]
    cdef bint flag
    #cdef int D = V.shape[0] - 1
    
    cdef np.ndarray[Df_t, ndim=1] y_n = np.zeros(N, dtype=Df)
    
    cdef double cur_y, cur_kj
    
    cdef int n, j, u, k, cat_c, raw_c
    
    
    for n in range(N):
        cur_y = w
        for j in range(J):
            for k in range(K):
                flag = False
                cur_kj = 1.
                cat_c = 0
                raw_c = 0
                for u in range(U):
                    if Z[u,j]==1:
                        flag = True
                        if u==raw[raw_c]:
                            cur_kj *= X_r[n,raw_c]*V[raw_i[raw_c],k]
                        else:
                            cur_kj *= V[X_c[n,cat_c],k]
                    if u!=raw[raw_c]:
                       if l_cat>cat_c:
                            cat_c += 1
                    elif l_raw>raw_c: 
                        raw_c += 1
                #if cur_kj != 1:
                if flag:
                    cur_y += cur_kj
        cat_c = 0
        raw_c = 0
        for u in range(U):
            if u==raw[raw_c]:
                cur_y += X_r[n,raw_c]*w_l[raw_i[raw_c]]
                if l_raw>raw_c:
                    raw_c += 1
            else:
                cur_y += w_l[X_c[n,cat_c]]
                if l_cat>cat_c:
                    cat_c += 1
#        if cur_y>0:
        y_n[n] = cur_y
    return y_n
    
#@cython.boundscheck(False) # turn off bounds-checking for entire function
#@cython.wraparound(False)
#@cython.cdivision(True)
#@cython.nonecheck(False)
#@cython.profile(False)
def true_y(data, double w, np.ndarray[Df_t, ndim=1] V, np.ndarray[Di_t, ndim=2] Z):
    assert V.dtype == Df and Z.dtype == Di
    #cdef np.ndarray[Di_t, ndim=2] Z = Z.copy()
    #y, X_c, X_r, cat, J, U, gamma, raw, H, raw_i = Z_param - pois model
    #y, X_c, X_r, cat, gamma, raw, H, raw_i = Z_param - gauss model
    cdef np.ndarray[Di_t, ndim=2] X_c = data[0]
    cdef np.ndarray[Df_t, ndim=2] X_r = data[1]
    #cdef np.ndarray[Di_t, ndim=1] cat = np.array(data[2])
    cdef np.ndarray[Di_t, ndim=1] raw_i = data[2]
    cdef np.ndarray[Di_t, ndim=1] raw = data[3]
    cdef int l_raw = raw.shape[0] - 1
    cdef int l_cat = X_c.shape[1] - 1
    
    cdef int J = Z.shape[1]
    cdef int U = Z.shape[0]
    cdef int N = X_r.shape[0]
    cdef int zind = data[4]
    cdef bint flag
    #cdef int D = V.shape[0] - 1
    
    cdef np.ndarray[Df_t, ndim=1] y_n = np.zeros(N, dtype=Df)
    
    cdef double cur_y, cur_kj
    
    cdef int n, j, u, cat_c, raw_c
    
    
    for n in range(N):
        cur_y = w
        for j in range(J):
            flag = False
            cur_kj = V[j]
            cat_c = 0
            raw_c = 0
            for u in range(U):
                if Z[u,j]==1:
                    flag = True
                    if u==raw[raw_c]:
                        cur_kj *= X_r[n,raw_c]
                    else:
                        cur_kj *= (X_c[n,cat_c]!=zind)
                if u!=raw[raw_c]:
                   if l_cat>cat_c:
                        cat_c += 1
                elif l_raw>raw_c: 
                    raw_c += 1
            #if cur_kj != 1:
            if flag:
                cur_y += cur_kj
        y_n[n] = cur_y
    return y_n