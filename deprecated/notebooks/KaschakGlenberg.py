#!/usr/bin/env python
# coding: utf-8

# # Kaschak and Glenberg
# 
# Try to replicate experiment 2 from Kaschack and Glenberg (2000) in LMs.

# In[1]:


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
import scipy

import src.sent_encoder

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# In[2]:


data = pd.read_csv("../data/kaschak-glenberg.csv")


# In[3]:


with open("../data/bnc.pkl", "rb") as f:
  bnc_data = pickle.load(f)


# ## Get contextual vectors for verbs

# In[4]:


LAYER = 11
enc = src.sent_encoder.SentEncoder()


# In[ ]:


vecs_ditransitive = enc.sentence_vecs(data.sent_ditransitive.tolist(), verbs=data.verb.tolist())[:, LAYER]
vecs_transitive = enc.sentence_vecs(data.sent_transitive.tolist(), verbs=data.verb.tolist())[:, LAYER]


# In[6]:


vecs_ditransitive.shape


# ## Average vector for "gave"

# In[7]:


gave_vec = enc.avg_contextual_word_vec(bnc_data, "gave")[LAYER]
gave_vec.shape


# ## Distance to "gave" for every contextual verb vector

# In[8]:


result_df = []
for ix, row in data.iterrows():
  result_df.append(pd.Series({
    'sent_ditransitive': row.sent_ditransitive,
    'sent_transitive': row.sent_transitive,
    'verb': row.verb,
    'gave_dist_ditransitive': np.linalg.norm(gave_vec - vecs_ditransitive[ix]),
    'gave_dist_transitive': np.linalg.norm(gave_vec - vecs_transitive[ix]),
    
    # Similar results if we use cosine instead of euclidean distance.
    #'gave_dist_ditransitive': scipy.spatial.distance.cosine(gave_vec, vecs_ditransitive[ix]),
    #'gave_dist_transitive': scipy.spatial.distance.cosine(gave_vec, vecs_transitive[ix]),
  }))

result_df = pd.DataFrame(result_df)


# ## Test significance
# 
# Result: ditransitive is slightly closer to "gave" than transitive, but difference is not significant.

# In[9]:


gave_delta = result_df.gave_dist_transitive - result_df.gave_dist_ditransitive
print("Average gave_dist_transitive - gave_dist_ditransitive:", np.mean(gave_delta))
print("Number of sentences where ditransitive closer to vec(gave):", np.sum(gave_delta > 0))
print("Number of sentences where transitive closer to vec(gave):", np.sum(gave_delta < 0))


# In[10]:


sns.histplot(gave_delta, bins=10)
plt.show()


# In[11]:


scipy.stats.ttest_rel(result_df.gave_dist_transitive, result_df.gave_dist_ditransitive)

