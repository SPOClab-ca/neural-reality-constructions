#!/usr/bin/env python
# coding: utf-8

# # Nonse Sentence Generation
# 
# Generate nonsense sentences similar to Johnson and Goldberg (2013)

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
from nltk.corpus import treebank

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# ## Obtain list of singular nouns, adjectives, past-tense verbs

# In[2]:


penn = nltk.corpus.reader.bracket_parse.BracketParseCorpusReader("../data/PTB3", ".*\.tree")


# In[3]:


len(penn.parsed_sents())


# In[4]:


singular_nouns = set()
adjectives = set()
past_verbs = set()

for tree in penn.parsed_sents():
  for leaf in tree.subtrees(lambda t: t.height() == 2):
    if leaf.label() == "NN":
      singular_nouns.add(leaf[0].lower())
    if leaf.label() == "JJ":
      adjectives.add(leaf[0].lower())
    if leaf.label() == "VBD":
      past_verbs.add(leaf[0].lower())


# In[5]:


print(len(singular_nouns))
print(len(adjectives))
print(len(past_verbs))

