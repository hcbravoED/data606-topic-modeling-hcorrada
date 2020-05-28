import numpy as np

# initialize parameters in the EM algorithm for pLSA
# 
# input
#   - num_words (int): number of words
#   - num_docs (int): number of docs
#   - num_topics (int): number of topics
#
# output (tuple of length 2):
#   [0] (array shape (num_topics, num_docs)): topic distirbution per doc
#   [1] (array shape (num_words, num_topics)): word distribution per topic
def init_params(num_words, num_docs, num_topics):
    log_p = np.zeros((num_topics, num_docs))
    log_theta = np.zeros((num_words, num_topics))
    
    # intialize p
    for d in range(num_docs):
        log_p[:, d] = np.log(np.random.dirichlet(np.ones((num_topics))))
    
    # intialize theta
    for t in range(num_topics):
        log_theta[:,t] = np.log(np.random.dirichlet(np.ones((num_words))))
        
    return log_p, log_theta

# Perform the estep in the EM algorithm for pLSA
# 
# overwrites input array gamma
#
# input:
#   - gamma (array shape (num_words, num_docs, num_topics)): word/doc/topic responsibilities
#   - x (array shape (num_words, num_docs)): word/doc number of occurrences
#   - log_p (array shape (num_topics, num_docs)): topic distribution per doc (in log space)
#   - log_theta (array shape (num_words, num_topics)): word distribution per topic (in log space)
#
# output:
#   NONE, gamma is overwritten
def estep(gamma, x, log_p, log_theta):
    nz_words, nz_docs = x.nonzero()
    for w, d in zip(nz_words,nz_docs):
        tmp = np.exp(log_p[:,d] + log_theta[w,:])
        denom = np.sum(tmp)
        gamma[w,d,:] = x[w,d] * tmp / denom
                    
# Perform the M step of EM algorithm for pLSA
#
# input:
#   - x (array of shape (num_words, num_docs)): number of word occurrences per doc
#   - gamma (array of shape (num_words, num_docs, num_topics)): word/document/topic responsibilities
#   - pseudo_count (float): avoid divide by 0
#
# output (tuple of size 2):
#   [0] (array of shape (num_topics, num_docs)): topic distribution per doc (in log space)
#   [1] (array of shape (num_words, num_topics)): word distirbution per topic (in log space)
def mstep(x, gamma, pseudo_count=0.01):
    num_words, num_docs, num_topics = gamma.shape
    
    res_p = np.transpose(np.sum(gamma, axis=0)) + pseudo_count
    for d in range(num_docs):
        res_p[:,d] = np.log(res_p[:,d]) - np.log(np.sum(res_p[:,d]))
            
    res_theta = np.sum(gamma, axis=1) + pseudo_count
    for t in range(num_topics):
        res_theta[:,t] = np.log(res_theta[:,t]) - np.log(np.sum(res_theta[:,t])) 
    
    return res_p, res_theta

# compute the log likelihood of given estimates
# 
# input:
#   - x (array of shape (num_words, num_docs)): number of word occurences per doc
#   - log_p (array of shape(num_topics, num_docs)): topic distribution for each doc (in log space)
#   - log_theta (array of shape(num_words, num_topics)): word distribution for each doc (in log space)
#
# output:
#    (float): log likelihood of model
def get_loglik(x, log_p, log_theta):
    res = 0
    nz_words, nz_docs = x.nonzero()
    for w, d in zip(nz_words, nz_docs):
        res += x[w, d] * np.log((np.sum(np.exp(log_p[:, d] + log_theta[w, :]))))
    return res
    
# computes value of Q function at given estimates
#
# input:
#   - x (array): word/document occurrence matrix
#   - gamma (array): word/document/topic responsabilities
#   - log_p (array): topic distribution estimate
#   - log_theta (array): word distribution estimate
#
# output:
#   (float): value of Q function at given estimates
def get_qopt(x, gamma, log_p, log_theta):
    res = 0.0
    nz_words, nz_docs = x.nonzero()
    for w, d in zip(nz_words, nz_docs):
        res += np.sum(gamma[w,d,:] * log_p[:,d])
        res += np.sum(gamma[w,d,:] * log_theta[w,:])
    return res

# check convergence of the EM algorithm
# 
# checks if we haven't passed the iteration limit
# checks if improvement in log likelihood is smaller than given tolerance
# 
# input:
#   - cur_it (int): the current iteration
#   - cur_llik (float): the current log likelihood
#   - new_llik (float): the new log likelihood after parameter update
#   - max_iter (int): maximum number of iterations
#   - eps (float): tolerance for log likelihood improvement
def check_convergence(cur_it, cur_llik, new_llik, max_iter, eps):
    return cur_it >= max_iter or ((new_llik - cur_llik) < eps)

# do one run of the EM algorithm for pLSA
#
# input:
#   - x (array of shape (num_words, num_docs)): number of word occurences per doc
#   - num_topics (int): number of topics to estimate in the pLSA model
#   - max_iter (int): maximum number of iterations
#   - eps: tolerance used when checking convergence
#   - verbose (bool): print convergence information per iteration
def plsa_em_one_run(x, num_topics=8, max_iter=20, eps=1e-3, verbose=False):
    num_words, num_docs = x.shape
    print_template = "It: {0:d}, loglik: {1:.5f}, old_q: {2:.5f}, new_q: {3:.5f}"

    # initialize parameters p and theta (in log-space)
    log_p, log_theta = init_params(num_words, num_docs, num_topics)    
    assert(log_p.shape == (num_topics, num_docs))
    assert(log_theta.shape == (num_words, num_topics))
    
    # matrix to store responsiblity values
    gamma = np.zeros((num_words, num_docs, num_topics))
    
    # compute the likelihood of the initialized parameters
    cur_llik = get_loglik(x, log_p, log_theta)
    
    i = 0
    while True:
        # perform the estep, overwrites the gamma array
        estep(gamma, x, log_p, log_theta)
        
        # compute the value of the Q function at current (p,theta) values
        old_q = get_qopt(x, gamma, log_p, log_theta)
        
        # perform the m step to return new p and theta values
        new_log_p, new_log_theta = mstep(x, gamma)
        assert(new_log_p.shape == (num_topics, num_docs))
        assert(new_log_theta.shape == (num_words, num_topics))
    
        # compute likelihood of new parameter estimates
        new_llik = get_loglik(x, new_log_p, new_log_theta)        
        
        # compute value of Q function at new (p, theta) values (should be higher)
        new_q = get_qopt(x, gamma, new_log_p, new_log_theta)
        
        if verbose:
            print(print_template.format(i, new_llik, old_q, new_q))
            
        # break if converged
        if check_convergence(i, cur_llik, new_llik, max_iter, eps):
            break

        i += 1
        log_p, log_theta, cur_llik = new_log_p, new_log_theta, new_llik
    return np.exp(log_p), np.exp(log_theta), cur_llik

# Estimate parameters of a pLSA model using the EM algorithm
#
# input:
#   - x (array of shape (num_words, num_docs)): number of word occurrences per doc
#   - num_restarts (int): number of EM runs to do with random initialization
#   - num_topics (int): number of topics to estimate in the pLSA model
#   - max_iter (int): number of maximum iterations to perform per EM run
#   - eps (float): tolerance used when checking convergence
#   - verbose (bool): print convergence information per iteration
#
# output (tuple of length 3):
#   [0] (array of shape (num_topics, num_docs)): the estimated topic distribution for each doc
#   [1] (array of shape (num_words, num_topics)): the estimated word distribution for each topic
#   [2] (float): the log-likelihood of the estimated parameters
def plsa_em(x, num_restarts = 3, num_topics=8, max_iter=20, eps=1e-3, verbose=False):
    num_words, num_docs = x.shape
    
    best_p = None
    best_theta = None
    best_llik = -np.inf
    
    for r in range(num_restarts):
        if verbose:
            print("Run {} of {}".format(r, num_restarts))
            
        p, theta, llik = plsa_em_one_run(x, num_topics, max_iter, eps, verbose)
        
        if llik > best_llik:
            best_p = p
            best_theta = theta
            best_llik = llik
    
    return best_p, best_theta, best_llik
            