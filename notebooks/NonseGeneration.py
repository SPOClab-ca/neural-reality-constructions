#!/usr/bin/env python
# coding: utf-8

# # Nonse Sentence Generation
# 
# Generate nonsense sentences similar to Johnson and Goldberg (2013)

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
import nltk
from nltk.corpus import treebank

import src.sent_encoder

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


singular_nouns = list(singular_nouns)
adjectives = list(adjectives)
past_verbs = list(past_verbs)

print(len(singular_nouns))
print(len(adjectives))
print(len(past_verbs))


# ## Get contextual vecs for gave/made/put/took

# In[6]:


with open("../data/bnc.pkl", "rb") as f:
  bnc_data = pickle.load(f)


# In[7]:


LAYER = 11
enc = src.sent_encoder.SentEncoder()


# In[8]:


prototype_vecs = {
  'gave': enc.avg_contextual_word_vec(bnc_data, "gave")[LAYER],
  'made': enc.avg_contextual_word_vec(bnc_data, "made")[LAYER],
  'put': enc.avg_contextual_word_vec(bnc_data, "put")[LAYER],
  'took': enc.avg_contextual_word_vec(bnc_data, "took")[LAYER],
}


# ## Generate sentences of each type

# In[9]:


random.seed(12345)
NUM_SENTENCES_PER_CXN = 1000
sentences = defaultdict(list)

# Ditransitive: S/he nonseV-ed him/her the nonseN
for i in range(NUM_SENTENCES_PER_CXN):
  pronoun1 = random.choice(["He", "She"])
  pronoun2 = random.choice(["him", "her"])
  nonse_verb = random.choice(past_verbs)
  nonse_noun = random.choice(singular_nouns)
  sentences['ditransitive'].append(f"{pronoun1} {nonse_verb} {pronoun2} the {nonse_noun}.")
  
# Resultative: S/he nonseV-ed it nonseAdj.
for i in range(NUM_SENTENCES_PER_CXN):
  pronoun1 = random.choice(["He", "She"])
  nonse_verb = random.choice(past_verbs)
  nonse_adj = random.choice(adjectives)
  sentences['resultative'].append(f"{pronoun1} {nonse_verb} it {nonse_adj}.")
  
# Caused-motion: S/he nonseV-ed it on the nonseN.
for i in range(NUM_SENTENCES_PER_CXN):
  pronoun1 = random.choice(["He", "She"])
  nonse_verb = random.choice(past_verbs)
  nonse_noun = random.choice(singular_nouns)
  sentences['caused-motion'].append(f"{pronoun1} {nonse_verb} it on the {nonse_noun}.")
  
# Removal: S/he nonseV-ed it from him/her.
for i in range(NUM_SENTENCES_PER_CXN):
  pronoun1 = random.choice(["He", "She"])
  pronoun2 = random.choice(["him", "her"])
  nonse_verb = random.choice(past_verbs)
  sentences['removal'].append(f"{pronoun1} {nonse_verb} it from {pronoun2}.")

