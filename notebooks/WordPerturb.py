#!/usr/bin/env python
# coding: utf-8

# # Auto Word Perturbation
# 
# Strategy: find k closest word(s) in GLoVe embedding space, limited to nouns that match the given word on singular/plural and capitalization. Assume that the given word is in vocab and a noun (but check and error out if this is not the case).

# In[2]:


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

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# In[3]:


model = KeyedVectors.load_word2vec_format("../data/glove.840B.300d.txt")


# In[6]:


model.most_similar("pig", topn=5)

