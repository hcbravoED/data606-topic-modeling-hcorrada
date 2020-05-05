import numpy as np

# generate a word/document/topic occurrence matrix given model parameters
# 
# input:
#   - num_words_per_doc (int): number of words per document
#   - p (array of shape (num_topics, num_docs)): probability distribution of topics in a document (colsums == 1)
#   - theta (array of shape (num_words, num_topics)): probability distribution of words in a topic (colsums == 1)
#
# output:
#   array of shape (num_words, num_docs, num_topics): matrix of number of words per document per topic 
def simulate_delta(num_words_per_doc, p, theta):
    # check that the shapes of are compatible
    assert(p.shape[0] == theta.shape[1])
    num_topics, num_docs = p.shape
    num_words = theta.shape[0]
    
    # this will store word-topic-document counts (unobserved) 
    delta = np.zeros((num_words, num_docs, num_topics))
    
    # finish this loop to populate delta
    for d in range(num_docs):
        pass
    
    return delta
        
# Simulate a dataset for topic modeling. Generates model parameters from a Dirichlet distribution
#
# input:
#   - num_words (int): number of words in corpus
#   - num_docs (int): number of documents in corpus
#   - num_topics (int): number of topics in LDA model from which data is generated
#   - num_words_per_doc (int): number of total number of words in a document
#
# output (tuple of length 4):
#   [0] (array of shape (num_words, num_docs)): number of occurrences of each word in each document (x)
#   [1] (array of shape (num_words, num_docs, num_topics)): number of occurrences of each word 
#               generated from each topic in each document (delta)
#   [2] (array of shape (num_topics, num_docs)): topic distribution for each document (p, colsums == 1)
#   [3] (array of shape (num_words, num_topics)): word distribution for each topic (theta, colsums == 1)
def simulate_data(num_words, num_docs, num_topics, num_words_per_doc):
    # generate topic distribution for each document
    p = np.zeros((num_topics, num_docs))
    
    # assign extra weight to a specific topic in each document
    # by using a non-uniform alpha parameter to the Dirichlet
    # distribution
    for d in range(num_docs):
        # initialize allpha vector to be uniform (1.)
        alpha = np.ones((num_topics))
        
        # rotate important topic across documents
        # and set alpha for that topic to be 10.
        t = d % num_topics
        alpha[t] = 10.
        
        p[:,d] = np.random.dirichlet(alpha)
        
    # generate word distribution for each topic
    theta = np.zeros((num_words, num_topics))
    alpha = np.ones((num_words, num_topics))
    
    # set some number of useful words (with high probability)
    # for each topic
    n_useful_words = 5 * num_topics
    
    for w in range(n_useful_words):
        # rotate the useful words across topics
        t = w % num_topics
        alpha[w, t] = 10.
        
    for t in range(num_topics):
        theta[:, t] = np.random.dirichlet(alpha[:,t])
        
    delta = simulate_delta(num_words_per_doc, p, theta)
    x = np.sum(delta, axis=2).reshape((num_words, num_docs))
    return x, delta, p, theta