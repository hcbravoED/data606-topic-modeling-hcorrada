import numpy as np
import pickle
import scipy.sparse
from pathlib import Path

from sklearn.datasets import fetch_20newsgroups

import gensim
from gensim.matutils import corpus2csc
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

import nltk

def prep_nltk():
    print("[prep_nltk] Downloading wordnet...")
    nltk.download('wordnet')

def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess_doc(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
    
    

def create_processed_docs(data_dir):
    docs_path = Path(data_dir + "/processed_docs.pkl")
    if docs_path.exists():
        print("Found stored processed docs file. Loading...")
        with open(docs_path, 'rb') as pkl_file:
            processed_docs = pickle.load(pkl_file)
        return processed_docs
    
    print("Did not find stored processed docs file. Creating...")
    newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True)

    targets_to_keep = [4, 9, 13]
    keep_article = np.isin(newsgroups_train.target, targets_to_keep)
    
    docs_to_process = np.array(newsgroups_train.data)[keep_article]
    processed_docs = []
    for doc in docs_to_process:
        processed_docs.append(preprocess_doc(doc))
        
    with open(docs_path, 'wb') as pkl_file:
        pickle.dump(processed_docs, pkl_file)
    return processed_docs

def create_dictionary(data_dir, processed_docs, no_below=15, no_above=0.1, keep_n=10000):
    dict_path = Path(data_dir + "/dictionary.pkl")
    if dict_path.exists():
        print("Found dictionary object. Loading...")
        with open(dict_path, 'rb') as pkl_file:
            dictionary = pickle.load(pkl_file)
            return dictionary
    
    print("Did not find dictionary object. Creating...")
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    print(dictionary)
    
    with open(dict_path, 'wb') as pkl_file:
        pickle.dump(dictionary, pkl_file)
        
    return dictionary


def create_docmat(data_dir, processed_docs, dictionary):        
    docmat_path = Path(data_dir + "/newsgroup_docmat.npz")
    if docmat_path.exists():
        print("Found saved docmat file. Loading...")
        docmat = scipy.sparse.load_npz(docmat_path)
        return docmat
    
    print("Did not find saved docmat file, generating...")
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    doc_mat = corpus2csc(corpus)
    scipy.sparse.save_npz(docmat_path, doc_mat)
    return doc_mat

def create_dataset(data_dir="data"):
    data_path = Path(data_dir)
    if not data_path.exists():
        print("Creating directory ", data_dir)
        data_path.mkdir()
        
    prep_nltk()
    processed_docs = create_processed_docs(data_dir)
    dictionary = create_dictionary(data_dir, processed_docs)
    return create_docmat(data_dir, processed_docs, dictionary)  

def get_docmat(data_dir="data"):
    docmat_path = Path(data_dir + "/newsgroup_docmat.npz")
    if not docmat_path.exists():
        print("Did not find stored docmat file. Creating docmat...")
        return create_dataset(data_dir)
        
    print("Found saved docmat file. Loading...")
    docmat = scipy.sparse.load_npz(docmat_path)
    return docmat

def print_important_words(theta, num_words=5, data_dir="data"):
    dict_path = Path(data_dir + "/dictionary.pkl")
    if not dict_path.exists():
        print("Did not find file with dictionary. Exiting...")
        return None
    
    with open(dict_path, 'rb') as pkl_file:
            dictionary = pickle.load(pkl_file)
            
    print(dictionary)
    _, num_topics = theta.shape 
    for t in range(num_topics):
        print([dictionary[i] for i in np.argsort(theta[:,t])[-num_words:]])