#!/usr/bin/env python
# coding: utf-8

# # Extract contructions from Penn treebank

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


# ## Read PTB Data

# In[2]:


penn = nltk.corpus.reader.bracket_parse.BracketParseCorpusReader("../data/PTB3", ".*\.tree")


# In[3]:


len(penn.parsed_sents())


# In[4]:


print(penn.parsed_sents()[0])


# ## Find LET-ALONE and WAY constructions

# In[5]:


def find_let_alone(tree):
  return "let alone" in " ".join(tree.leaves())

for tree in penn.parsed_sents():
  if find_let_alone(tree):
    print(" ".join(tree.leaves()))


# In[6]:


# Find trees containing something like (NP (PRP$ his) (NN way))
def find_way(tree):
  for st in tree.subtrees():
    if len(st) == 2 and st[0].label().startswith("PRP") and st[1].leaves() == ["way"]:
      return True
  return False

for tree in penn.parsed_sents():
  if find_way(tree):
    print(" ".join(tree.leaves()))

