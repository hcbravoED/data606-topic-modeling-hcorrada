import numpy as np

# initialize parameters of LDA model
#
# input:
#   - num_words: number of words
#   - num_docs: number of documents
#   - num_topics: number of topics
# 
# output (tuple of length 2):
#   [0] (array shape(num_topics, num_docs)): topic distribution for each doc
#   [1] (array shape(num_words, num_topics)): word distribution for each topic
def init_params(num_words, num_docs, num_topics):
    # TODO: finish this function
    log_p = np.zeros((num_topics, num_docs))
    log_theta = np.zeros((num_words, num_topics))
    return log_p, log_theta

# compute topic probability for each word/doc
# 
# Note: overwrites the topic_probs parameter
#
# input:
#   - topic_probs (array shape (num_words, num_docs, num_topics)): current topic probabilities for each word/doc
#   - x (array shape (num_words, num_docs)): word/doc occurrence counts
#   - log_p (array shape(num_topic, num_docs)): topic distribution for each doc (in log space)
#   - log_theta (array shape(num_words, num_topics)): word distribution for each topic (in log space)
def compute_topic_probs(topic_probs, x, log_p, log_theta):
    nz_words, nz_docs = x.nonzero()
    for w,d in zip(nz_words, nz_docs):
        # TODO: finish this loop
        pass

# reassign word occurrences across topics
#
# Note: this should rewrite delta matrix
#
# input:
#   - delta (array (num_words, num_docs, num_topics)): current sample of word/doc/topic counts
#   - x (array (num_words, num_docs)): word/doc occurrence count matrix
#   - topic_probs (array (num_words, num_docs, num_topics)): topic probability for each word/doc
#
# output:
#   NONE - overwrites delta parameter
def reassign_words(delta, x, topic_probs):
    nz_words, nz_docs = x.nonzero()
    for w, d in zip(nz_docs, nz_words):
        # TODO: finish this loop
        pass

# resample topic distributions for each document
# 
# input:
#   - delta: sample of word/doc/topic counts
#   - alpha: parameter of the Dirichlet prior
#
# output:
#   (array shape (num_topics, num_docs)): topic distribution for each document
def resample_p(delta, alpha):
    _, num_docs, num_topics = delta.shape
    log_p = np.zeros((num_topics, num_docs))
    # TODO: finish this function
    return log_p

# resample word distributions for each topic
# 
# input:
#   - delta: sample of word/doc/topic counts
#   - beta: parameter of the Dirichlet prior
#
# output:
#   (matrix shape (num_words, num_topics)): sample of word distribution for each topic
def resample_theta(delta, beta):
    num_words, _, num_topics = delta.shape
    log_theta = np.zeros((num_words, num_topics))
    # TODO: finish this function
    return log_theta


# Gibbs sampler for the LDA topic model
#
# input
#   - x (array shape (num_words, num_docs)): number of word occurrences per document
#   - num_topics (int): number of topics in LDA model
#   - num_rounds (int): number of total rounds of sampling
#   - burnin_fraction (float): fraction of sampling rounds used for burn in
#   - alpha (float): parameter in Dirichlet prior for p (topic distirbution per document)
#   - beta (float): parameter in Dirichlet prior of theta (word distribution per topic)
#   - verbose (bool): print iteration count
#
# output (tuple of length 2):
#   [0] (array shape (num_topics, num_docs)): topic distribution for each document
#   [1] (array shape (num_words, num_topics)): word distribution for each topic
def lda_gibbs(x, num_topics=8, num_rounds=200, burnin_fraction=.2, alpha=1., beta=1., verbose=False):
    num_words, num_docs = x.shape
    
    # figure out how many samples to use in burn in
    num_burnin = int(num_rounds * burnin_fraction)
    num_samples = num_rounds - num_burnin
    
    # initialize parameters
    log_p, log_theta = init_params(num_words, num_docs, num_topics)
    
    # matrix to store word/doc/topic probabilities
    topic_probs = np.zeros((num_words, num_docs, num_topics))
    
    # matrix to store word/doc/topic count samples
    delta = np.zeros((num_words, num_docs, num_topics))
    
    # matrix to store running sum of word/doc/topic count samples
    delta_sum = np.zeros((num_words, num_docs, num_topics))
                         
    i = 0
    while i < num_rounds:
        # update the word/doc/topic probabilities
        # this overwrites matrix topic_probs1
        compute_topic_probs(topic_probs, x, log_p, log_theta)
        
        # reassign word occurrences to topics based on probabilities
        # this overwrites matrix delta
        reassign_words(delta, x, topic_probs)
        
        # add delta to running sum if we are beyond burnin
        if i > num_burnin:
            delta_sum += delta
            
        # resample topic distributions for each document
        log_p = resample_p(delta, alpha)
        
        # resample word distribution for each topic
        log_theta = resample_theta(delta, alpha)
        
        if verbose and i % 10 == 0:
            print("Iteration {}".format(i))
        i += 1
        
    # done with samples, compute the mean word/doc/topic counts
    delta_hat = delta_sum / num_samples
    
    # compute final topic distributions for each document
    # TODO: compute this from delta_hat
    p_hat = np.zeros((num_topics, num_docs))
    
    # compute final word distributions for each topic
    # TODO: compute this from delta_hat
    theta_hat = np.zeros((num_words, num_topics))
    return p_hat, theta_hat