#!/usr/bin/env python
# coding: utf-8

# # Auto Word Perturbation
# 
# Strategy: find k closest word(s) in GLoVe embedding space, limited to nouns that match the given word on singular/plural and capitalization. Assume that the given word is in vocab and a noun (but check and error out if this is not the case).

# In[7]:


import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import torch
from collections import defaultdict
import random
import math
import pickle
from gensim.models import KeyedVectors
import nltk

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# In[3]:


model = KeyedVectors.load_word2vec_format("../data/glove.840B.300d.txt")


# In[6]:


model.most_similar("pig", topn=5)


# ## Get set of singular and plural nouns from Brown corpus

# In[16]:


nouns_sg = set()
nouns_pl = set()

for w, pos in nltk.corpus.brown.tagged_words():
  if pos == 'NN':
    nouns_sg.add(w.lower())
  if pos == 'NNS':
    nouns_pl.add(w.lower())


# In[21]:


print(len(nouns_sg))
print(len(nouns_pl))

assert 'women' in nouns_pl
assert 'women' not in nouns_sg
assert 'sheep' in nouns_sg
assert 'sheep' in nouns_pl


# In[41]:


def capitalize(w):
  return w[0].upper() + w[1:]
  
def closest_matching_words(w, topn=5):
  is_caps = w[0].isupper()
  w = w.lower()
  
  w2v_similar = model.most_similar(w, topn=100)
  ans = []
  for sim_w, _ in w2v_similar:
    if w in nouns_sg and sim_w in nouns_sg or         w in nouns_pl and sim_w in nouns_pl:
      ans.append(capitalize(sim_w) if is_caps else sim_w)
  return ans[:topn]


# In[42]:


closest_matching_words("pig")


# In[43]:


closest_matching_words("pigs")


# In[44]:


closest_matching_words("Pigs")

