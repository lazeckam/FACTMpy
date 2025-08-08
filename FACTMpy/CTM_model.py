"""
This module provides Correlated Topic Model model with all the nodes.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import beta, gamma, digamma, gammaln
from .utils import *
EPS = 1e-20

class nodeCTM_general():
    """
    Class to store const. of a CTM model.
    """
    def __init__(self, N, L, G, I, J):
        self.L = L
        self.N = N
        self.G = G
        self.I = I
        self.J = J


class nodeCTM_Sigma0(nodeCTM_general):
    """
    Class to define Sigma0 node (L times L).
    """
    def __init__(self, Sigma0, general_params):
        super().__init__(**general_params)

        self.Sigma0 = Sigma0

        self.inv_Sigma0 = np.linalg.inv(self.Sigma0)
        self.det_Sigma0 = np.linalg.det(self.Sigma0)
    
    def MB(self, eta_node, w_z_node, mu0_node):
        self.eta_node = eta_node
        self.w_z_node = w_z_node
        self.mu0_node = mu0_node
    
    def update(self):
        centered_mean = self.eta_node.vi_mu  - self.mu0_node.mu0 - self.w_z_node.E_w_z

        # Below we compute: diag(Cov(\sum_k z_nk w_.k, \sum_k' z_nk' w_.k'))
        # for l!= l' Cov(w_lk, w_l'k), so just diag needed
        Ez2w2 = np.sum(np.dot(self.w_z_node.E_z_squared, self.w_z_node.E_w_squared.T), axis=0)
        Ezw_2 = np.sum(np.dot(self.w_z_node.E_z**2, self.w_z_node.E_w.T**2), axis=0)
        cov_sumk_znk_wk = Ez2w2 - Ezw_2

        self.Sigma0 = np.dot(centered_mean.T, centered_mean)/self.N + np.diag(np.mean(self.eta_node.vi_var, axis=0)) + np.diag(cov_sumk_znk_wk)/self.N          

        self.inv_Sigma0 = np.linalg.inv(self.Sigma0)
        self.det_Sigma0 = np.linalg.det(self.Sigma0)


class nodeCTM_mu0(nodeCTM_general):
    """
    Class to define mu0 node (length L).
    """
    def __init__(self, mu0, general_params):
        super().__init__(**general_params)

        self.mu0 = mu0

    def MB(self, eta_node, w_z_node):
        self.eta_node = eta_node
        self.w_z_node = w_z_node

    def update(self):
        self.mu0 = np.mean(self.eta_node.vi_mu - self.w_z_node.E_w_z, axis=0)

class nodeCTM_w_z(nodeCTM_general):
    """
    Class to store variational params from FA part of FACTM.
    """
    def __init__(self, E_w, E_w_squared, E_z, E_z_squared, general_params):
        super().__init__(**general_params)

        self.E_w = E_w
        self.E_w_squared = E_w_squared

        self.E_z = E_z
        self.E_z_squared = E_z_squared

        self.E_w_z = np.dot(self.E_z, self.E_w.T)
        self.E_w_z_squared = np.dot(self.E_z**2, (self.E_w.T)**2)

    def MB(self):
        pass

    def update(self):
        pass

    def ELBO(self):
        pass


class nodeCTM_eta(nodeCTM_general):
    """
    Class to define eta node (N times L).
    """
    def __init__(self, vi_mu, vi_var, vi_zeta, general_params):
        super().__init__(**general_params)

        self.vi_mu = vi_mu
        self.vi_var = vi_var
        self.vi_zeta = vi_zeta

        self.E_eta_minus_mu0 = self.vi_mu
        self.E_exp_eta = np.exp(self.vi_mu + self.vi_var/2)

        self.elbo = 0

    def MB(self, w_z_node, mu0_node, Sigma0_node, xi_node):
        self.w_z_node = w_z_node
        self.mu0_node = mu0_node
        self.Sigma0_node = Sigma0_node
        self.xi_node = xi_node

    def update(self):

        for n in range(self.N):

            self.vi_zeta[n] = np.sum(self.E_exp_eta[n,:])

            f = lambda x: -self.__f_eta_par_n(x[:self.L], x[self.L:], 
                                    self.w_z_node.E_w_z[n,:], self.mu0_node.mu0, self.Sigma0_node.inv_Sigma0,
                                    self.xi_node.vi_par[n], 
                                    self.vi_zeta[n], self.I[n])

        
            fgrad = lambda x: -self.__fgrad_eta_par_n(x[:self.L], x[self.L:], 
                                            self.w_z_node.E_w_z[n,:], self.mu0_node.mu0, self.Sigma0_node.inv_Sigma0,
                                            self.xi_node.vi_par[n], 
                                            self.vi_zeta[n], self.I[n])
        
            starting_point = np.concatenate((1.0*self.vi_mu[n,:], 1.0*self.vi_var[n,:]))    

            # a condition that variances are non-negative
            bnds = tuple(map(lambda x : (None, None) if x < self.L else (EPS, None), range(2*self.L)))

            result = minimize(f, x0=starting_point, method='L-BFGS-B',  jac=fgrad,  options={'disp': 0, 'maxiter': 25}, bounds=bnds)


            self.vi_mu[n,:] = result.x[:self.L]
            self.vi_var[n,:] = result.x[self.L:]

            self.E_exp_eta[n,:] = np.exp(self.vi_mu[n,:] + self.vi_var[n,:]/2)
        self.E_eta_minus_mu0 = self.vi_mu - self.mu0_node.mu0

    def ELBO(self):

        elbo = np.sum(np.log(self.vi_var))/2 + self.N*self.L/2

        centered_mean = self.vi_mu - self.w_z_node.E_w_z - self.mu0_node.mu0

        # Below we compute: diag(Cov(\sum_k z_nk w_.k, \sum_k' z_nk' w_.k'))
        # for l!= l' Cov(w_lk, w_l'k), so just diag needed
        Ez2w2 = np.sum(np.dot(self.w_z_node.E_z_squared, self.w_z_node.E_w_squared.T), axis=0)
        Ezw_2 = np.sum(np.dot(self.w_z_node.E_z**2, self.w_z_node.E_w.T**2), axis=0)
        cov_sum_znk_wk = Ez2w2 - Ezw_2
        
        elbo += - self.N*np.log(self.Sigma0_node.det_Sigma0)/2 - np.sum(np.diag(self.Sigma0_node.inv_Sigma0)*self.vi_var)/2 \
            - np.sum(centered_mean * np.dot(centered_mean, self.Sigma0_node.inv_Sigma0))/2 \
            - np.dot(cov_sum_znk_wk, np.diag(self.Sigma0_node.inv_Sigma0))/2
        
        self.elbo = elbo

    def __f_eta_par_n(self, vi_eta_mu_n, vi_eta_var_n,E_w_z, mu, Sigma_inv,
                      vi_xi_par_n, vi_zeta_n, I):
    
        term_xi = np.sum(vi_xi_par_n, axis=0)

        return -np.sum(vi_eta_var_n*np.diag(Sigma_inv))/2 \
            - np.sum(np.dot(vi_eta_mu_n - E_w_z - mu, Sigma_inv)*(vi_eta_mu_n - E_w_z - mu))/2 \
                +np.sum(vi_eta_mu_n*term_xi) - (I/vi_zeta_n)*np.sum(np.exp(vi_eta_mu_n + vi_eta_var_n/2)) \
                + np.sum(np.log(vi_eta_var_n))/2
    
    def __fgrad_eta_par_n(self, vi_eta_mu_n, vi_eta_var_n,
                    E_w_z, mu, Sigma_inv,
                    vi_xi_par_n, 
                    vi_zeta_n, I):
    
        term_xi = np.sum(vi_xi_par_n, axis=0)
        term_MGF = (I/vi_zeta_n)*(np.exp(vi_eta_mu_n + vi_eta_var_n/2))
        
        grad_mu = - np.dot(Sigma_inv, vi_eta_mu_n - E_w_z - mu) + term_xi - term_MGF
        
        grad_var = - np.diag(Sigma_inv)/2 - term_MGF/2 + 1/(2*vi_eta_var_n)

        grad = np.concatenate((grad_mu, grad_var))

        return grad


class nodeCTM_xi(nodeCTM_general):
    """
    Class to define xi node (a list of N elements: I_n times L).
    """
    def __init__(self, vi_par, general_params):
        super().__init__(**general_params)

        self.vi_par = vi_par

        self.vi_log_par = [log_eps(vi_par[i]) for i in range(len(vi_par))]

        self.elbo = 0

    def MB(self, eta_node, beta_node, data):
        self.eta_node = eta_node
        self.beta_node = beta_node
        self.y_node = data

    def update(self):

        term_E_log_beta = self.beta_node.digamma_vi_alpha.T - self.beta_node.digamma_sum_vi_alpha

        for n in range(self.N):

            vi_par_n = np.zeros((self.I[n], self.L))
            vi_log_par_n = np.zeros((self.I[n], self.L))
            
            vi_log_par_n = self.eta_node.vi_mu[n,:] + np.dot(self.y_node.data[n], term_E_log_beta)
            vi_log_par_n = vi_log_par_n - np.outer(np.max(vi_log_par_n, axis=1), np.ones(self.L))
            vi_par_n = np.exp(vi_log_par_n)

            norm_cons_tmp = np.outer(np.sum(vi_par_n, axis = 1), np.ones(self.L))
            vi_par_n = vi_par_n/norm_cons_tmp
            vi_log_par_n = vi_log_par_n - log_eps(norm_cons_tmp)

            self.vi_log_par[n] = vi_log_par_n
            self.vi_par[n] = vi_par_n

    def ELBO(self):

        kl = 0
        entropy = 0
        for n in range(self.N):
            entropy += -np.sum(self.vi_par[n] * self.vi_log_par[n])
            kl += np.sum(self.eta_node.vi_mu[n,:] * self.vi_par[n]) \
                - self.I[n]*(np.log(self.eta_node.vi_zeta[n]) + np.sum(self.eta_node.E_exp_eta[n,:])/self.eta_node.vi_zeta[n] - 1)
            
        self.elbo = kl + entropy


class nodeCTM_beta(nodeCTM_general):
    """
    Class to define beta node (L times G).
    """
    def __init__(self, alpha, vi_alpha, general_params):
        super().__init__(**general_params)

        # N x L x G
        self.alpha = alpha
        self.vi_alpha = vi_alpha

        self.lnGamma_sum_vi_alpha = gammaln(np.sum(self.vi_alpha, axis=1))
        self.sum_lnGamma_alpha = self.G*gammaln(self.alpha)

        self.lnGamma_sum_alpha = gammaln(self.G*self.alpha)
        self.sum_lnGamma_vi_alpha = np.sum(gammaln(self.vi_alpha), axis=1)

        self.digamma_vi_alpha = digamma(self.vi_alpha)
        self.digamma_sum_vi_alpha = digamma(np.sum(self.vi_alpha, axis=1))

        self.elbo = 0

    def MB(self, xi_node, y_node):
        self.xi_node = xi_node
        self.y_node = y_node

    def update(self):

        vi_alpha = self.alpha*np.ones((self.L, self.G))

        for n in range(self.N):
            vi_alpha += np.dot(self.xi_node.vi_par[n].T, self.y_node.data[n])
        
        self.vi_alpha = vi_alpha

        sum_alpha_tmp = np.sum(vi_alpha, axis=1)

        self.lnGamma_sum_vi_alpha = gammaln(sum_alpha_tmp)
        self.sum_lnGamma_vi_alpha = np.sum(gammaln(vi_alpha), axis=1)

        self.digamma_vi_alpha = digamma(self.vi_alpha)
        self.digamma_sum_vi_alpha = digamma(sum_alpha_tmp)

    def ELBO(self):
        
        elbo = np.sum(self.lnGamma_sum_vi_alpha) - self.L*self.lnGamma_sum_alpha  \
                - np.sum(self.sum_lnGamma_vi_alpha)+ self.L*self.sum_lnGamma_alpha \
                    + np.sum((self.vi_alpha - self.alpha)*(self.digamma_vi_alpha.T - self.digamma_sum_vi_alpha).T)
        
        self.elbo = -elbo


class nodeCTM_y(nodeCTM_general):
    """
    Class to define observed y node (a list of N elements: I_n times G).
    """
    def __init__(self, data, general_params):
        super().__init__(**general_params)

        self.data = data

        self.elbo = 0

    def MB(self, beta_node, xi_node):
        self.beta_node = beta_node
        self.xi_node = xi_node

    def ELBO(self):

        elbo = 0
        for n in range(self.N):
            term_E_log_beta = (self.beta_node.digamma_vi_alpha.T  - self.beta_node.digamma_sum_vi_alpha)
            elbo += np.sum(self.xi_node.vi_par[n]*np.dot(self.data[n], term_E_log_beta))
        
        self.elbo = elbo


def starting_params_Sigma(starting_params, L):

    if 'Sigma' in starting_params.keys():
        Sigma = 1*starting_params['Sigma']
    else:
        Sigma = np.eye(L)

    return Sigma


def starting_params_mu(starting_params, L):

    if 'mu' in starting_params.keys():
        mu = 1*starting_params['mu']
    else:
        mu = np.zeros(L)

    return mu


def starting_params_beta(starting_params, L, G):

    if 'topics' in starting_params.keys():
        topics = 1*starting_params['topics']
        # print("beta")
    else:
        # par=100*1 so the distribution is close to uniform but not uniform
        topics = np.random.dirichlet(100*np.ones(G), size=L)

    return topics


class CTM():
    """
    Class to define Correlated Topic Model.
    """
    def __init__(self, data, N, L, G, K, starting_params = None, FA=True, *args, **kwargs):
        
        self.N = N
        self.L = L
        self.G = G

        # If FA=True we use FACTM, if FA=False simple CTM is fitted
        self.FA = FA

        if starting_params is None:
            starting_params = {}

        # compute I and J from data: 
        I = []
        J = []
        init_xi_par = []

        for n in range(N):
            data_n = data[n]

            I_n = data_n.shape[0]
            J_n = np.sum(data_n, axis=1)
            
            I.append(I_n)
            J.append(J_n)

            init_xi_par.append(np.ones((I_n, L))/L)

        self.I = I
        self.J = J

        # general params
        general_params = {'N': self.N, 'L': self.L, 'G': self.G, 'I': self.I, 'J': self.J}
        
        # CREATING NODES:
        # Sigma0
        Sigma0 = starting_params_Sigma(starting_params, self.L)
        self.node_Sigma0 = nodeCTM_Sigma0(Sigma0, general_params=general_params)

        # mu0
        mu0 = starting_params_mu(starting_params, self.L)
        self.node_mu0 = nodeCTM_mu0(mu0, general_params=general_params) 

        # beta
        topics = starting_params_beta(starting_params, self.L, self.G)
        self.node_beta = nodeCTM_beta(1e-5, topics, general_params=general_params)

        # W & Z
        if self.FA:
            self.node_w_z = nodeCTM_w_z(np.ones((L, K)), np.ones((L, K)), np.ones((N, K)), np.ones((N, K)),
                                        general_params=general_params)
        else:
            self.node_w_z = nodeCTM_w_z(np.zeros((L, K)), np.zeros((L, K)), np.zeros((N, K)), np.zeros((N, K)), 
                                        general_params=general_params)

        # eta
        self.node_eta = nodeCTM_eta(np.random.normal(size=(N, L))/10, np.ones((N, L))/10, np.ones(N), general_params=general_params)

        # xi
        self.node_xi = nodeCTM_xi(init_xi_par, general_params=general_params)

        # y
        self.node_y = nodeCTM_y(data, general_params=general_params)

        # ELBO
        self.elbo = 0

    def MB(self):

        self.node_mu0.MB(self.node_eta, self.node_w_z)
        self.node_Sigma0.MB(self.node_eta, self.node_w_z, self.node_mu0)
        self.node_beta.MB(self.node_xi, self.node_y)
        self.node_eta.MB(self.node_w_z, self.node_mu0, self.node_Sigma0, self.node_xi)
        self.node_xi.MB(self.node_eta, self.node_beta, self.node_y)
        self.node_y.MB(self.node_beta, self.node_xi)

    def update(self):

        self.node_xi.update()
        self.node_eta.update()
        self.node_beta.update()

        self.node_mu0.update()
        self.node_Sigma0.update()
        
    def ELBO(self):

        self.node_xi.ELBO()
        self.node_eta.ELBO()
        self.node_beta.ELBO()
        self.node_y.ELBO()

        elbo = self.node_xi.elbo + self.node_eta.elbo + self.node_beta.elbo + self.node_y.elbo

        self.elbo = elbo

    def get_elbo(self):

        return self.elbo
