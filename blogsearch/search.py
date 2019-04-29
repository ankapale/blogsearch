from django.http import HttpResponse
from django.shortcuts import render_to_response
from django.views.decorators import csrf
from django.views.decorators.csrf import csrf_exempt
import os
import pickle
from random import shuffle, seed
import numpy as np
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings
import sys  
 
# ### Sequence
# 1. fetch the list of text files or TF-IDF (Document corpus)
# 2. Train TF-IDF model (multiple versions may be) save the transformed data for each file
# 3. Store the mapping of the file to index and vice versa.
# 4. Store vocab and idf values (just in case)
# 5. Calculate the nearest neighbors and store
# 6. Get query, format it or do query expansion etc
# 7. transform to TF-IDF
# 8. Do dot product for cosine similarity
# 9. Sort the scosine similarities, get the best indexes
# 10. Based on the indexes and the index to file mapping get the filenames

# In[20]:


class Config(object):
    sim_dict_path = os.path.join(settings.BASE_DIR, "sim_dict.p")
    tf_idf_path = os.path.join(settings.BASE_DIR, "tf_idf.p")
    vectorizer_path = os.path.join(settings.BASE_DIR,"vectorizer.p")
    i2f_path =os.path.join(settings.BASE_DIR, "i2f.p")


# In[17]:


def train_tf_idf(file_list, **kwargs):
    # Default params
    tf_idf_params = {
        'input': 'filename', 
        'encoding': 'utf-8',
        'decode_error': 'replace',
        'strip_accents': 'unicode', 
        'lowercase': True,
        'analyzer': 'word',
        'stop_words': 'english', 
        'token_pattern': r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
        'ngram_range': (1, 2),
        'max_features':  5000, 
        'norm': 'l2',
        'use_idf': True,
        'smooth_idf': True,
        'sublinear_tf': True,
        'max_df': 1.0,
        'min_df': 1}
    
    # Update with kwargs if any
    tf_idf_params.update(kwargs)
    
    train_list = list(file_list) # creates copy
    shuffle(train_list)
    
    # compute tfidf vectors with scikits
    vectorizer = TfidfVectorizer(**tf_idf_params)
    vectorizer.fit(train_list)
    tf_idf_matrix = vectorizer.fit_transform(file_list)
    
    # CHange input type to content (string) dfor later
    vectorizer.input = 'content'
    
    return vectorizer, tf_idf_matrix

def vectorize_query(qraw, vectorizer):
    # apply any formatting if needed
    qraw = qraw.lower().strip() # lower case 
    # TODO: query expansion
    q_list = [qraw]
    return  vectorizer.transform(q_list)

    
def read_file(filename="blogname_url_list.txt"):
    with open(os.path.join(settings.BASE_DIR, filename), "r") as file:
        lines = file.read().splitlines()
    return lines

# create an iterator object to conserve memory
def make_corpus(paths):
    for p in paths:
        with open(os.path.join(settings.BASE_DIR, p), 'rb') as f:
            txt = f.read()
    yield txt

def cosine_ranking(tf_idf_matrix, query_vector, idx_to_file):
    scores = cosine_similarity(tf_idf_matrix, query_vector).flatten()
    score_tuples = [(idx_to_file[i], s) for i, s in enumerate(scores) if s > 0]
    score_tuples.sort(reverse=True, key=lambda a: a[1])
    return score_tuples

def nearest_neighbors(X, i2f):
    print("precomputing nearest neighbor queries in batches...")
    X = X.todense() # originally it's a sparse matrix
    sim_dict = {}
    batch_size = 200
    for i in range(0,len(X),batch_size):
        i1 = min(len(X), i+batch_size)
        xquery = X[i:i1] # BxD
        ds = -np.asarray(np.dot(X, xquery.T)) #NxD * DxB => NxB
        IX = np.argsort(ds, axis=0) # NxB
        for j in range(i1-i):
            sim_dict[i2f[i+j]] = [i2f[q] for q in list(IX[:50,j])]
        print('%d-%d/%d...' % (i, i1, len(X)))
    return sim_dict

def search_query(query, vectorizer, tf_idf_matrix, idx_to_file, ranking="cosine"):
    a = vectorize_query(query, vectorizer)
    if ranking == "cosine":
        return cosine_ranking(tf_idf_matrix, a, idx_to_file)

def search(query,  num_results=20, file_list="list_of_good_files.txt", path="./"):
    if os.path.isfile(Config.vectorizer_path) and os.path.isfile(Config.tf_idf_path) and os.path.isfile(Config.i2f_path):
        temp=os.path.join(settings.BASE_DIR,"vectorizer.p")
        temp2=open(temp, "rb")
        vectorizer = pickle.load(temp2)
        tf_idf_matrix = pickle.load(open(Config.tf_idf_path, "rb"))
        idx_to_file = pickle.load(open(Config.i2f_path, "rb"))
    else:
        file_list = read_file(file_list)
        file_list = [path + file for file in file_list]
        # Generate idx to file mapping
        idx_to_file = {i:k for i, k in enumerate(file_list)}

        vectorizer, tf_idf_matrix = train_tf_idf(file_list)
        sim_dict = nearest_neighbors(tf_idf_matrix, idx_to_file)

        pickle.dump(tf_idf_matrix, open(Config.tf_idf_path, "wb"))
        pickle.dump(vectorizer, open(Config.vectorizer_path, "wb"))
        pickle.dump(sim_dict, open(Config.sim_dict_path, "wb"))
        pickle.dump(idx_to_file, open(Config.i2f_path, "wb"))

    return search_query(query, vectorizer, tf_idf_matrix, idx_to_file)[:num_results]



def search_form(request):
    return render_to_response('search_form.html')
 

def search_get(request):  
    request.encoding='utf-8'
    if 'q' in request.GET:
        message = search(request.GET['q'], 20, "list_of_good_files.txt")
    else:
        message = search("Reinforement learning", 20, "list_of_good_files.txt")
    return HttpResponse(message)