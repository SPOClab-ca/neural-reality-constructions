#!/usr/bin/env python
# coding: utf-8

# # Filter Eligible Sentences from FrameNet
# 
# Find sentences that are possibly useful for perturbation.
# 
# * Length of sentence must be <= 10 tokens. (16k sentences remaining)
# * One of the frame elements must be <= 3 tokens and contain a noun. (10k sentences remaining)

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


# In[2]:


def approx_len(sent):
  return len(sent.split())

fn_sents = fn.exemplars()
short_sentences = [sent for sent in fn_sents if approx_len(sent.text) <= 10]


# In[3]:


len(short_sentences)


# ## Filter sentences with a noun in one of its frame elements

# In[4]:


def is_overlapping(start1, end1, start2, end2):
  return (start1 <= start2 and end1 > start2) or (start2 <= start1 and end2 > start1)

print(is_overlapping(1, 2, 3, 4)) #False
print(is_overlapping(1, 2, 2, 4)) #False
print(is_overlapping(1, 2, 1, 4)) #True
print(is_overlapping(1, 4, 1, 2)) #True
print(is_overlapping(1, 4, 2, 3)) #True
print(is_overlapping(1, 4, 4, 5)) #False
print(is_overlapping(1, 4, 3, 5)) #True
print(is_overlapping(1, 4, 0, 1)) #False
print(is_overlapping(1, 4, 0, 5)) #True
print(is_overlapping(1, 4, 0, 3)) #True


# In[5]:


def pos_is_noun(pos_tag):
  return 'nn' in pos_tag.lower()

print(pos_is_noun('NN1')) #True
print(pos_is_noun('PROPN')) #False


# In[6]:


def has_noun_target(sent):
  for fe_start, fe_end, fe_name in sent.FE[0]:
    fe_text = sent.text[fe_start:fe_end]
    if approx_len(fe_text) <= 3:
      for pos_start, pos_end, pos_tag in sent.POS:
        if pos_is_noun(pos_tag) and is_overlapping(fe_start, fe_end, pos_start, pos_end):
          return True
  return False

noun_target_sents = [sent for sent in short_sentences if has_noun_target(sent)]


# In[7]:


len(noun_target_sents)

