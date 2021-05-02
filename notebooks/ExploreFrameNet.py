#!/usr/bin/env python
# coding: utf-8

# # Explore FrameNet 1.7

# In[1]:


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
import nltk
from nltk.corpus import framenet as fn

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# ## Look at some annotated sentences

# In[2]:


fn_sents = fn.sents()


# In[3]:


fn_sents[12345]


# ## Distribution of sentence lengths

# In[4]:


counter = defaultdict(int)
for sent in fn_sents:
  ntok = len(nltk.tokenize.word_tokenize(sent.text))
  counter[ntok] += 1


# In[5]:


plt.figure(figsize=(15, 5))
plt.bar(counter.keys(), counter.values())
plt.xlim((0, 120))
plt.show()


# ## Look at some shorter sentences

# In[12]:


short_sentences = [sent for sent in fn_sents[:200] if len(nltk.tokenize.word_tokenize(sent.text)) < 10]


# In[13]:


[sent.text for sent in short_sentences]

