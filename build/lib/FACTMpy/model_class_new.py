from tqdm import tqdm
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from .FACTM_model_new import *


class FACTModel(FACTM):
    def __init__(self, data, K, L, likelihoods, 
                 Z_priors=None, W_priors=None, seed=None, *args, **kwargs):

        if seed is not None:
            self.seed = seed
            np.random.seed(self.seed)

        self.K = K
        self.L = L
        self.data = data

        self.__assign_params(likelihoods, Z_priors, W_priors)        

        super(FACTModel, self).__init__(data, self.N, self.M, self.K, self.D, self.G, 
                                        self.likelihoods, self.Z_priors, self.W_priors)

        self.__first_fit = True

        self.elbo_sequence = []
        print('new')

    def pretrain(self, FA_pretrain='PCA', CTM_pretrain='CTM'):

        if FA_pretrain not in ['PCA', 'FA']:
            raise TypeError('Pretraning for FA part should be one of: PCA, FA')
        if CTM_pretrain not in ['CTM']:
            raise TypeError('Pretraning for CTM part should be one of: CTM')

        if CTM_pretrain == 'CTM':
            for m in range(self.M):
                if self.likelihoods[m] == 'CTM':
                    print('Pretraning CTM for a modality ' + str(m))

                    mod_ctm_tmp = CTM(self.ctm_list['M'+str(m)].node_y.data, self.N, self.ctm_list['M'+str(m)].L,
                                      self.ctm_list['M'+str(m)].G, self.fa.K, 
                                      FA=False)
                    
                    mod_ctm_tmp.MB()

                    for i in range(50):
                        mod_ctm_tmp.update()

                    mod_ctm_tmp.FA = True
                    self.ctm_list['M'+str(m)] = mod_ctm_tmp

                    self.fa.nodelist_y[m].data = self.ctm_list['M'+str(m)].node_eta.vi_mu - self.ctm_list['M'+str(m)].node_mu0.mu0

        print('Pretraning FA')
        if FA_pretrain == 'PCA':
            modFA = PCA(n_components=self.fa.K, whiten=True)
        if FA_pretrain == 'FA':
            modFA = FactorAnalysis(n_components=self.fa.K)

        # scaled and centered data
        if CTM_pretrain == 'CTM':
            data_tmp = []
            for m in range(self.M):
                data_tmp_m = self.fa.nodelist_y[m].data
                data_tmp.append((data_tmp_m - np.nanmean(data_tmp_m, axis=0))/np.nanstd(data_tmp_m, axis=0))
            data_tmp = np.hstack(data_tmp)

        D_tmp = [self.D[m] for m in range(self.M)]

        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        data_tmp = imp.fit_transform(data_tmp)
        modFA.fit(data_tmp)

        views_segments = [0] + np.cumsum(np.array(D_tmp)).tolist()
        loadings_tmp = modFA.components_.T
        latent_factors_tmp = modFA.transform(data_tmp)

        for m in range(self.M):
            # get weights + scale back according to the variance of the features
            self.fa.nodelist_hat_w[m].vi_mu = (np.std(self.fa.nodelist_y[m].data, axis=0)*loadings_tmp[views_segments[m]:views_segments[m+1],:].T).T

        # scale to [-1, 1] as in MOFA
        min_max_scaler = MinMaxScaler((-1, 1))
        self.fa.node_z.vi_mu = min_max_scaler.fit_transform(latent_factors_tmp)


    def fit(self, pretrain=True, max_iter=1000, elbo_tres=0):

        if self.__first_fit:
            self.MB()

            # pretrain
            if pretrain:
                pretrain()

            # elbo at start
            # update to make sure all the params make sense
            self.update()
            self.ELBO()
            self.elbo_sequence.append(self.get_elbo())

        self.__first_fit = False
        
        print('Fitting a model.')

        for iter in tqdm(range(max_iter - 1)):
            self.update()
            self.ELBO()
            self.elbo_sequence.append(self.get_elbo())

            if self.elbo_sequence[-1] - self.elbo_sequence[-2] < elbo_tres:
                break

    # Point estimators
    def get_latent_factors(self):
        return self.fa.node_z.vi_mu
    
    def get_loadings_dense(self, m):
        return self.fa.nodelist_hat_w[m].vi_mu

    def get_loadings_sparse(self, m):
        return self.fa.nodelist_w[m].E_w
    
    def get_featurewise_sparsity(self, m):
        return self.fa.nodelist_s[m].vi_gamma
    
    def get_mu0(self, m):
        return self.ctm_list['M'+str(m)].node_mu0.mu0
    
    def get_Sigma0(self, m):
        return self.ctm_list['M'+str(m)].node_Sigma0.Sigma0
    
    def get_topics(self, m):
        L_m = self.L_M[np.where(np.array(self.index_CTM) == m)[0][0]]
        return np.array([self.ctm_list['M'+str(m)].node_beta.vi_alpha[l,:]/np.sum(self.ctm_list['M'+str(m)].node_beta.vi_alpha[l, :]) for l in range(L_m)])
    
    def get_eta(self, m):
        return self.ctm_list['M'+str(m)].node_eta.vi_mu
    
    def get_eta_probabilities_of_topics(self, m):
        L_m = self.L_M[np.where(np.array(self.index_CTM) == m)[0][0]]
        prob_est = np.exp(self.get_pe_eta(m))
        prob_est = prob_est/np.outer(np.sum(prob_est, axis=1), np.ones(L_m))
        return prob_est
    
    def get_probabilities_of_topics(self, m):
        return self.ctm_list['M'+str(m)].node_xi.vi_par
    
    def get_clusters(self, m):
        return [np.argmax(self.ctm_list['M'+str(m)].node_xi.vi_par[n], axis=1) for n in range(self.N)]   
    
    def get_predictions_FA(self, m):
        if self.O[m]:
            return self.fa.nodelist_w[m].E_w_z
        else:
            return np.dot(self.fa.node_z.vi_mu, self.fa.nodelist_w[m].E_w.T)
    
    def __assign_params(self, likelihoods, Z_priors, W_priors):
        # M
        self.M = len(self.data)

        # N
        if likelihoods[0] == 'CTM':
            # if M0 a structered view
            self.N = len(self.data['M0'])
        else:
            # if  M0 a simple view
            self.N = self.data['M0'].shape[0]

        # D, G and the number of structered views
        D = []
        G = []

        m_ctm = 0
        for m in range(self.M):
            if not likelihoods[m] == 'CTM':
                D.append(self.data['M'+str(m)].shape[1])
            else:
                D.append(self.L[m_ctm])
                m_ctm += 1
                G.append(self.data['M'+str(m)][0].shape[1])

        self.M_CTM = m_ctm
        self.D = D
        self.G = G

        self.likelihoods = likelihoods
        if Z_priors is None:
            self.Z_priors = ['stdN']*self.K
        if W_priors is None:
            self.W_priors = [] 
            for m in range(self.M):
                if not likelihoods[m] == 'CTM':
                    self.W_priors.append('ARD')
                else:
                    self.W_priors.append('ARD')
            


