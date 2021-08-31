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

import src.sent_encoder

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# In[2]:


data = pd.read_csv("../data/kaschak-glenberg.csv")


# ## Get contextual vectors for verbs

# In[3]:


enc = src.sent_encoder.SentEncoder()


# In[ ]:


vecs_ditransitive = enc.sentence_vecs(data.sent_ditransitive.tolist(), verbs=data.verb.tolist())
vecs_transitive = enc.sentence_vecs(data.sent_transitive.tolist(), verbs=data.verb.tolist())


# In[7]:


vecs_ditransitive.shape

