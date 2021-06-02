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
import json
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

assert not is_overlapping(1, 2, 3, 4)
assert not is_overlapping(1, 2, 2, 4)
assert is_overlapping(1, 2, 1, 4)
assert is_overlapping(1, 4, 1, 2)
assert is_overlapping(1, 4, 2, 3)
assert not is_overlapping(1, 4, 4, 5)
assert is_overlapping(1, 4, 3, 5)
assert not is_overlapping(1, 4, 0, 1)
assert is_overlapping(1, 4, 0, 5)
assert is_overlapping(1, 4, 0, 3)


# In[5]:


# Filters for common (non-proper) nouns in PENN and BNC tagsets
def pos_is_noun(pos_tag):
  return 'nn' in pos_tag.lower() and 'np' not in pos_tag.lower()

assert not pos_is_noun('VB')
assert pos_is_noun('NN1')
assert not pos_is_noun('NNPS')
assert not pos_is_noun('NN1-NP0')


# In[6]:


def get_noun_targets(sent):
  noun_targets = []
  for fe_start, fe_end, fe_name in sent.FE[0]:
    fe_text = sent.text[fe_start:fe_end]
    if approx_len(fe_text) <= 3:
      for pos_start, pos_end, pos_tag in sent.POS:
        if pos_is_noun(pos_tag) and is_overlapping(fe_start, fe_end, pos_start, pos_end):
          noun_targets.append(sent.text[pos_start:pos_end])
  return noun_targets

noun_target_sents = []
for sent in short_sentences:
  noun_targets = get_noun_targets(sent)
  if noun_targets != []:
    noun_target_sents.append((sent, noun_targets))


# In[7]:


len(noun_target_sents)


# ## Export random selection to CSV

# In[8]:


random.seed(12345)
export_sents = random.sample(noun_target_sents, 100)


# In[9]:


df = []
for sent, noun_targets in export_sents:
  annotations = []
  annotations.append(('Target', sent.text[sent.Target[0][0]:sent.Target[0][1]]))
  for span_start, span_end, fe_name in sent.FE[0]:
    annotations.append((fe_name, sent.text[span_start:span_end]))
    
  df.append(pd.Series({
    'frame': sent.frame.name,
    'text': sent.text,
    'noun_targets': json.dumps(noun_targets),
    'annotations': json.dumps(annotations),
  }))
df = pd.DataFrame(df)


# In[10]:


df.to_csv("short_fn_exemplars.csv", index=False, encoding='utf-8')

