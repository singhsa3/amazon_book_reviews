import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import multiprocessing
from multiprocessing import Pool
from multiprocessing import cpu_count
import re


def parallelize(df, func):
    cores = cpu_count()-5
    partitions = cores
    df_split = np.array_split(df, partitions)
    pool = Pool(cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def stop(example_sent):
    example_sent = example_sent.lower()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(example_sent)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    wordnet_lemmatizer = WordNetLemmatizer()
    ''' punctuations = r"[\w']+|[.,!?;]"  
    
    for word in filtered_sentence:
        if word in punctuations:
            filtered_sentence.remove(word)
    Decided against punctiations '''
    filtered_Lemm_sentence = [wordnet_lemmatizer.lemmatize(w) for w in filtered_sentence ]
    return filtered_Lemm_sentence

def word_func(data):
    print("Process working on: ", data)
    return data.apply(stop)

def doc_model(alldocs):
    from gensim.models import Doc2Vec
    import gensim.models.doc2vec
    from collections import OrderedDict
    import multiprocessing

    cores = multiprocessing.cpu_count()
    assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

    simple_models = [
        # PV-DBOW plain
        Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample=0,
                epochs=20, workers=cores),
        # PV-DM w/ default averaging; a higher starting alpha may improve CBOW/PV-DM modes
        Doc2Vec(dm=1, vector_size=300, window=10, negative=5, hs=0, min_count=2, sample=0,
                epochs=20, workers=cores, alpha=0.05, comment='alpha=0.05'),
    ]
    for model in simple_models:
        model.build_vocab(alldocs)
        print("%s vocabulary scanned & state initialized" % model)

    return simple_models


def my_predict(example_sent,model):
    test_sample = stop(example_sent)
    # Convert the sample document into a list and use the infer_vector method to get a vector representation for it
    new_doc_vec = model.infer_vector(test_sample, steps=200, alpha=0.05)
    # use the most_similar utility to find the most similar documents.
    return model.docvecs.most_similar(positive=[new_doc_vec])


