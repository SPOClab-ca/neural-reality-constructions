#!/usr/bin/env python
# coding: utf-8

# # Perek (2012) construction templates
# 
# Generate 4x4 design stimuli sets similar to Perek (2012).

# In[1]:


import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import torch
from collections import defaultdict
import random
import math
import pickle

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# In[2]:


MALE_NAMES = [
  "Aaron",
  "Adam",
  "Andrew",
  "Bill",
  "Bob",
  "Bruce",
  "Charlie",
  "Chris",
  "David",
  "Don",
  "Doug",
  "Dylan",
  "Eddie",
  "Edward",
  "Eric",
  "Frank",
  "Fred",
  "George",
  "Harry",
  "Henry",
  "Jack",
  "James",
  "Jason",
  "Jerry",
  "Joe",
  "John",
  "Joseph",
  "Mark",
  "Michael",
  "Mike",
  "Paul",
  "Sam",
  "Steve",
  "Thomas",
  "Tom",
  "William",
]


# In[3]:


LIQUID_NOUNS = [
  "acid",
  "beer",
  "bleach",
  "blood",
  "coffee",
  "cream",
  "gasoline",
  "gin",
  "honey",
  "ink",
  "juice",
  "milk",
  "oil",
  "paint",
  "sand",
  "shampoo",
  "slime",
  "soda",
  "soup",
  "syrup",
  "tea",
  "vodka",
  "water",
  "whisky",
  "wine",
  "yogurt",
]


# In[4]:


DITRANSITIVE_VERBS = [
  "bought",
  "brought",
  "handed",
  "gave",
  "lent",
  "offered",
  "passed",
  "served",
  "sent",
  "threw",
]

TRANSITIVE_VERBS = [
  "brought",
  "chucked",
  "gave",
  "handed",
  "kicked",
  "offered",
  "passed",
  "sent",
  "threw",
  "tossed",
]

CAUSED_MOTION_VERBS = [
  "dropped",
  "dumped",
  "poured",
  "spilled",
  "splashed",
  "sprayed",
  "sprinkled",
]

WITH_APPLICATIVE_VERBS = [
  "brushed",
  "coated",
  "covered",
  "dabbed",
  "plastered",
  "rubbed",
  "sprayed",
  "stained",
]


# ## Randomized generation

# In[5]:


random.seed(12345)

def gen_stimuli():
  # Some set difference operations to avoid duplicate verbs
  ditransitive_verbs = random.sample(DITRANSITIVE_VERBS, 4)
  transitive_verbs = random.sample(set(TRANSITIVE_VERBS) - set(ditransitive_verbs), 4)
  caused_motion_verbs = random.sample(CAUSED_MOTION_VERBS, 4)
  with_applicative_verbs = random.sample(set(WITH_APPLICATIVE_VERBS) - set(caused_motion_verbs), 4)
  
  names = random.sample(MALE_NAMES, 32)
  nouns = random.sample(LIQUID_NOUNS, 16)
  
  sentences = []
  
  # Ditransitive
  sentences.append({
    "sentence": f"{names[0]} {ditransitive_verbs[0]} {names[1]} some {nouns[0]}.",
    "construction": "ditransitive"
  })
  sentences.append({
    "sentence": f"{names[2]} {ditransitive_verbs[1]} {names[3]} some {nouns[1]}.",
    "construction": "ditransitive"
  })
  sentences.append({
    "sentence": f"{names[4]} {ditransitive_verbs[2]} {names[5]} some {nouns[2]}.",
    "construction": "ditransitive"
  })
  sentences.append({
    "sentence": f"{names[6]} {ditransitive_verbs[3]} {names[7]} some {nouns[3]}.",
    "construction": "ditransitive"
  })
  
  # Transitive
  sentences.append({
    "sentence": f"{names[8]} {transitive_verbs[0]} some {nouns[4]} to {names[9]}.",
    "construction": "transitive"
  })
  sentences.append({
    "sentence": f"{names[10]} {transitive_verbs[1]} some {nouns[5]} to {names[11]}.",
    "construction": "transitive"
  })
  sentences.append({
    "sentence": f"{names[12]} {transitive_verbs[2]} some {nouns[6]} to {names[13]}.",
    "construction": "transitive"
  })
  sentences.append({
    "sentence": f"{names[14]} {transitive_verbs[3]} some {nouns[7]} to {names[15]}.",
    "construction": "transitive"
  })
  
  # Caused motion
  sentences.append({
    "sentence": f"{names[16]} {caused_motion_verbs[0]} some {nouns[8]} on {names[17]}.",
    "construction": "caused-motion"
  })
  sentences.append({
    "sentence": f"{names[18]} {caused_motion_verbs[1]} some {nouns[9]} on {names[19]}.",
    "construction": "caused-motion"
  })
  sentences.append({
    "sentence": f"{names[20]} {caused_motion_verbs[2]} some {nouns[10]} on {names[21]}.",
    "construction": "caused-motion"
  })
  sentences.append({
    "sentence": f"{names[22]} {caused_motion_verbs[3]} some {nouns[11]} on {names[23]}.",
    "construction": "caused-motion"
  })
  
  # With applicative
  sentences.append({
    "sentence": f"{names[24]} {with_applicative_verbs[0]} {names[25]} with some {nouns[12]}.",
    "construction": "with-applicative"
  })
  sentences.append({
    "sentence": f"{names[26]} {with_applicative_verbs[1]} {names[27]} with some {nouns[13]}.",
    "construction": "with-applicative"
  })
  sentences.append({
    "sentence": f"{names[28]} {with_applicative_verbs[2]} {names[29]} with some {nouns[14]}.",
    "construction": "with-applicative"
  })
  sentences.append({
    "sentence": f"{names[30]} {with_applicative_verbs[3]} {names[31]} with some {nouns[15]}.",
    "construction": "with-applicative"
  })
  
  return pd.DataFrame(sentences)


# In[6]:


all_stimuli = []
for i in range(100):
  df = gen_stimuli()
  df['group'] = i
  all_stimuli.append(df)
all_stimuli = pd.concat(all_stimuli)


# In[7]:


all_stimuli[['group', 'sentence', 'construction']].to_csv("templated-stimuli.csv", index=False)

