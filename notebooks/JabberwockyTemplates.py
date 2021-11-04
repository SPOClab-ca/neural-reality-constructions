#!/usr/bin/env python
# coding: utf-8

# # Jabberwocky Templates
# 
# Generate Jabberwocky sentences similar to Johnson and Goldberg (2013)

# In[1]:


import pandas as pd
from collections import defaultdict
import random
import nltk
from nltk.corpus import treebank

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


# ## Generate sentences of each type

# In[7]:


random.seed(12345)
NUM_SENTENCES_PER_CXN = 5000
templated_sentences = []

# Ditransitive: S/he nonseV-ed him/her the nonseN
for i in range(NUM_SENTENCES_PER_CXN):
  pronoun1 = random.choice(["He", "She"])
  pronoun2 = random.choice(["him", "her"])
  nonse_verb = random.choice(past_verbs)
  nonse_noun = random.choice(singular_nouns)
  templated_sentences.append(pd.Series({
    'construction': 'ditransitive',
    'sentence': f"{pronoun1} {nonse_verb} {pronoun2} the {nonse_noun}.",
    'nonse_verb': nonse_verb,
  }))
  
# Resultative: S/he nonseV-ed it nonseAdj.
for i in range(NUM_SENTENCES_PER_CXN):
  pronoun1 = random.choice(["He", "She"])
  nonse_verb = random.choice(past_verbs)
  nonse_adj = random.choice(adjectives)
  templated_sentences.append(pd.Series({
    'construction': 'resultative',
    'sentence': f"{pronoun1} {nonse_verb} it {nonse_adj}.",
    'nonse_verb': nonse_verb,
  }))
  
# Caused-motion: S/he nonseV-ed it on the nonseN.
for i in range(NUM_SENTENCES_PER_CXN):
  pronoun1 = random.choice(["He", "She"])
  nonse_verb = random.choice(past_verbs)
  nonse_noun = random.choice(singular_nouns)
  templated_sentences.append(pd.Series({
    'construction': 'caused-motion',
    'sentence': f"{pronoun1} {nonse_verb} it on the {nonse_noun}.",
    'nonse_verb': nonse_verb,
  }))
  
# Removal: S/he nonseV-ed it from him/her.
for i in range(NUM_SENTENCES_PER_CXN):
  pronoun1 = random.choice(["He", "She"])
  pronoun2 = random.choice(["him", "her"])
  nonse_verb = random.choice(past_verbs)
  templated_sentences.append(pd.Series({
    'construction': 'removal',
    'sentence': f"{pronoun1} {nonse_verb} it from {pronoun2}.",
    'nonse_verb': nonse_verb,
  }))

templated_sentences = pd.DataFrame(templated_sentences)


# In[8]:


templated_sentences.to_csv("jabberwocky-templated.csv", index=False)

