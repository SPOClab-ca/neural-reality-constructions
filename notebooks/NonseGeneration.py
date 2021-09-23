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
import scipy

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


singular_nouns = defaultdict(int)
adjectives = defaultdict(int)
past_verbs = defaultdict(int)

for tree in penn.parsed_sents():
  for leaf in tree.subtrees(lambda t: t.height() == 2):
    if leaf.label() == "NN":
      singular_nouns[leaf[0].lower()] += 1
    if leaf.label() == "JJ":
      adjectives[leaf[0].lower()] += 1
    if leaf.label() == "VBD":
      past_verbs[leaf[0].lower()] += 1


# In[5]:


# Filter out words that are too low-frequency
singular_nouns = dict(filter(lambda w: w[1] >= 10, singular_nouns.items()))
adjectives = dict(filter(lambda w: w[1] >= 10, adjectives.items()))
past_verbs = dict(filter(lambda w: w[1] >= 10, past_verbs.items()))


# In[6]:


singular_nouns = list(sorted(singular_nouns))
adjectives = list(sorted(adjectives))
past_verbs = list(sorted(past_verbs))

print(len(singular_nouns))
print(len(adjectives))
print(len(past_verbs))


# ## Get contextual vecs for gave/made/put/took

# In[7]:


with open("../data/bnc.pkl", "rb") as f:
  bnc_data = pickle.load(f)


# In[8]:


LAYER = 11
enc = src.sent_encoder.SentEncoder()


# In[ ]:


prototype_vecs = {
  'gave': enc.avg_contextual_word_vec(bnc_data, "gave")[LAYER],
  'made': enc.avg_contextual_word_vec(bnc_data, "made")[LAYER],
  'put': enc.avg_contextual_word_vec(bnc_data, "put")[LAYER],
  'took': enc.avg_contextual_word_vec(bnc_data, "took")[LAYER],
}


# ## Alternate vecs for "gave"

# In[10]:


perek_data = pd.read_csv("../data/perek-templated.csv")

gave_ditransitive_sents = [s for s in perek_data[perek_data.construction == 'ditransitive'].sentence if "gave" in s]
gave_caused_motion_sents = [s for s in perek_data[perek_data.construction == 'to-dative'].sentence if "gave" in s]


# In[11]:


prototype_vecs = {
  'gave-bnc': enc.avg_contextual_word_vec(bnc_data, "gave")[LAYER],
  'gave-ditransitive': enc.avg_contextual_word_vec(gave_ditransitive_sents, "gave")[LAYER],
  'gave-caused-motion': enc.avg_contextual_word_vec(gave_caused_motion_sents, "gave")[LAYER],
  'gave-balanced': enc.avg_contextual_word_vec(gave_ditransitive_sents + gave_caused_motion_sents, "gave")[LAYER],
  'gave-decontextual': enc.contextual_token_vecs(["gave"])[1][0][0][LAYER],
}


# ## Generate sentences of each type

# In[12]:


random.seed(12345)
NUM_SENTENCES_PER_CXN = 5000
templated_sentences = defaultdict(list)

# Ditransitive: S/he nonseV-ed him/her the nonseN
for i in range(NUM_SENTENCES_PER_CXN):
  pronoun1 = random.choice(["He", "She"])
  pronoun2 = random.choice(["him", "her"])
  nonse_verb = random.choice(past_verbs)
  nonse_noun = random.choice(singular_nouns)
  templated_sentences['ditransitive'].append((
    f"{pronoun1} {nonse_verb} {pronoun2} the {nonse_noun}.",
    nonse_verb
  ))
  
# Resultative: S/he nonseV-ed it nonseAdj.
for i in range(NUM_SENTENCES_PER_CXN):
  pronoun1 = random.choice(["He", "She"])
  nonse_verb = random.choice(past_verbs)
  nonse_adj = random.choice(adjectives)
  templated_sentences['resultative'].append((
    f"{pronoun1} {nonse_verb} it {nonse_adj}.",
    nonse_verb
  ))
  
# Caused-motion: S/he nonseV-ed it on the nonseN.
for i in range(NUM_SENTENCES_PER_CXN):
  pronoun1 = random.choice(["He", "She"])
  nonse_verb = random.choice(past_verbs)
  nonse_noun = random.choice(singular_nouns)
  templated_sentences['caused-motion'].append((
    f"{pronoun1} {nonse_verb} it on the {nonse_noun}.",
    nonse_verb
  ))
  
# Removal: S/he nonseV-ed it from him/her.
for i in range(NUM_SENTENCES_PER_CXN):
  pronoun1 = random.choice(["He", "She"])
  pronoun2 = random.choice(["him", "her"])
  nonse_verb = random.choice(past_verbs)
  templated_sentences['removal'].append((
    f"{pronoun1} {nonse_verb} it from {pronoun2}.",
    nonse_verb
  ))


# ## Get distances from cxn-verbs to proto-verbs

# In[13]:


verb_dist_results = []

for cxn_type, cxn_sentences_and_verbs in templated_sentences.items():
  cxn_sentences = [t[0] for t in cxn_sentences_and_verbs]
  cxn_verbs = [t[1] for t in cxn_sentences_and_verbs]
  cxn_verb_vecs = enc.sentence_vecs(cxn_sentences, cxn_verbs)[:, LAYER]
  
  for proto_verb, proto_verb_vec in prototype_vecs.items():
    for i, cxn_verb_vec in enumerate(cxn_verb_vecs):
      dist = np.linalg.norm(proto_verb_vec - cxn_verb_vec)
      #dist = scipy.spatial.distance.cosine(proto_verb_vec, cxn_verb_vec)
      verb_dist_results.append(pd.Series({
        'cxn_sentence': cxn_sentences[i],
        'cxn': cxn_type,
        'verb': proto_verb,
        'dist': dist,
      }))
      
verb_dist_results = pd.DataFrame(verb_dist_results)


# ## Summarize results

# In[15]:


for verb in prototype_vecs.keys():
  for cxn in templated_sentences.keys():
    m = verb_dist_results[(verb_dist_results.cxn == cxn) & (verb_dist_results.verb == verb)].mean()
    sd = verb_dist_results[(verb_dist_results.cxn == cxn) & (verb_dist_results.verb == verb)].std()
    print(cxn, verb, float(m), float(sd))

