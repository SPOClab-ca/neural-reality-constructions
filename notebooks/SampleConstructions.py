#!/usr/bin/env python
# coding: utf-8

# # Sample Constructions
# 
# Explore using RoBERTa to determine correct vs incorrect on 2 sample constructions types (Caused-Motion and Way), 72 sentence pairs.

# In[1]:


import sys
sys.path.append('../')

import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import torch
import transformers
from transformers import pipeline

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# ## Load data
# 
# Keep only sentence pairs where the rating difference is >= 2

# In[2]:


data = pd.read_csv("../data/paired_sentences.csv")


# In[3]:


data = data[abs(data.Rating1 - data.Rating2) >= 2]


# In[4]:


data.head()


# ## Run RoBERTa
# 
# This function is copy-pasted from the layerwise anomaly project, but has some serious limitations:
# 
# 1. Assumes sentences have the same length
# 2. Assumes only one masked token differs
# 
# Therefore the majority of sentences are excluded right now...

# In[5]:


nlp = pipeline("fill-mask", model='roberta-base')


# In[6]:


def fill_one(sent1, sent2):
  toks1 = nlp.tokenizer(sent1, add_special_tokens=False)['input_ids']
  toks2 = nlp.tokenizer(sent2, add_special_tokens=False)['input_ids']
  
  if len(toks1) != len(toks2):
    return None

  masked_toks = []
  dtok1 = None
  dtok2 = None
  num_masks = 0
  for ix in range(len(toks1)):
    if toks1[ix] != toks2[ix]:
      masked_toks.append(nlp.tokenizer.mask_token_id)
      num_masks += 1
      dtok1 = toks1[ix]
      dtok2 = toks2[ix]
    else:
      masked_toks.append(toks1[ix])
  
  if num_masks != 1:
    return None
  
  res = nlp(nlp.tokenizer.decode(masked_toks), targets=[nlp.tokenizer.decode(dtok1), nlp.tokenizer.decode(dtok2)])
  return res[0]['token'] == dtok1


# In[7]:


for _, row in data.iterrows():
  result = fill_one(row['Sentence1'], row['Sentence2'])
  if result is None:
    continue
  print(row['Sentence1'])
  print(row['Sentence2'])
  print(f'Sentence {1 if result else 2} is better')
  print()

