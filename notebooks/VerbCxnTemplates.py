#!/usr/bin/env python
# coding: utf-8

# # Verb construction templates
# 
# Generate 4x4 design stimuli sets similar to Bencini and Goldberg (2000).

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
  "Adam",
  "Andrew",
  "Bill",
  "Bob",
  "David",
  "Eric",
  "Frank",
  "George",
  "Harry",
  "Henry",
  "Jack",
  "James",
  "John",
  "Michael",
  "Mike",
  "Paul",
  "Sam",
  "Steve",
  "Thomas",
  "Tom",
]


# In[3]:


# (verb past tense, direct object noun, resultative complement)
VERB_OBJ_RESULT = [
  ("pushed", "door", "shut"),
  ("pushed", "window", "open"),
  ("pushed", "laptop", "shut"),
  ("pushed", "fridge", "open"),
  ("kicked", "ball", "free"),
  ("kicked", "door", "open"),
  ("kicked", "box", "apart"),
  ("kicked", "wall", "down"),
  ("punched", "wall", "down"),
  ("punched", "door", "shut"),
  ("punched", "table", "apart"),
  ("punched", "book", "apart"),
  ("got", "fridge", "open"),
  ("got", "book", "closed"),
  ("got", "ball", "stuck"),
  ("got", "door", "open"),
  ("threw", "laptop", "apart"),
  ("threw", "window", "open"),
  ("threw", "box", "open"),
  ("threw", "book", "open"),
  ("sliced", "ball", "apart"),
  ("sliced", "book", "apart"),
  ("sliced", "bread", "apart"),
  ("sliced", "box", "apart"),
  ("cut", "ball", "open"),
  ("cut", "bread", "apart"),
  ("cut", "tree", "down"),
  ("cut", "box", "open"),
  ("pulled", "tree", "down"),
  ("pulled", "ball", "free"),
  ("pulled", "window", "open"),
  ("pulled", "door", "open"),
  ("hit", "wall", "down"),
  ("hit", "window", "open"),
  ("hit", "fridge", "open"),
  ("hit", "door", "open"),
  ("tore", "bread", "apart"),
  ("tore", "laptop", "apart"),
  ("tore", "wall", "down"),
  ("tore", "tree", "down"),
]


# In[4]:


PREPOSITION_PHRASES = [
  "into the house",
  "into the water",
  "into the plane",
  "into the net",
  "onto the roof",
  "onto the elevator",
  "onto the bed",
  "through the air",
  "through the hole",
  "through the forest",
  "to the hospital",
  "to the airport",
  "to the beach",
  "over the fence",
  "over the balcony",
  "off the bus",
  "off the bridge",
]


# In[5]:


VERBS = list(set([t[0] for t in VERB_OBJ_RESULT]))


# In[6]:


random.seed(12345)

def shuffled_complements(v):
  return random.sample([t for t in VERB_OBJ_RESULT if t[0] == v], 4)

def gen_stimuli():
  verbs = random.sample(VERBS, 4)
  complements = [shuffled_complements(v) for v in verbs]
  preposition_phrases = random.sample(PREPOSITION_PHRASES, 4)
  names = random.sample(MALE_NAMES, 20)
  
  sentences = []
  
  # Transitive
  sentences.append({
    "sentence": f"{names[0]} {verbs[0]} the {complements[0][0][1]}.",
    "verb": verbs[0],
    "construction": "transitive"
  })
  sentences.append({
    "sentence": f"{names[1]} {verbs[1]} the {complements[1][0][1]}.",
    "verb": verbs[1],
    "construction": "transitive"
  })
  sentences.append({
    "sentence": f"{names[2]} {verbs[2]} the {complements[2][0][1]}.",
    "verb": verbs[2],
    "construction": "transitive"
  })
  sentences.append({
    "sentence": f"{names[3]} {verbs[3]} the {complements[3][0][1]}.",
    "verb": verbs[3],
    "construction": "transitive"
  })
  
  # Ditransitive
  sentences.append({
    "sentence": f"{names[4]} {verbs[0]} {names[5]} the {complements[0][1][1]}.",
    "verb": verbs[0],
    "construction": "ditransitive"
  })
  sentences.append({
    "sentence": f"{names[6]} {verbs[1]} {names[7]} the {complements[1][1][1]}.",
    "verb": verbs[1],
    "construction": "ditransitive"
  })
  sentences.append({
    "sentence": f"{names[8]} {verbs[2]} {names[9]} the {complements[2][1][1]}.",
    "verb": verbs[2],
    "construction": "ditransitive"
  })
  sentences.append({
    "sentence": f"{names[10]} {verbs[3]} {names[11]} the {complements[3][1][1]}.",
    "verb": verbs[3],
    "construction": "ditransitive"
  })
  
  # Caused Motion
  sentences.append({
    "sentence": f"{names[12]} {verbs[0]} the {complements[0][2][1]} {preposition_phrases[0]}.",
    "verb": verbs[0],
    "construction": "caused-motion"
  })
  sentences.append({
    "sentence": f"{names[13]} {verbs[1]} the {complements[1][2][1]} {preposition_phrases[1]}.",
    "verb": verbs[1],
    "construction": "caused-motion"
  })
  sentences.append({
    "sentence": f"{names[14]} {verbs[2]} the {complements[2][2][1]} {preposition_phrases[2]}.",
    "verb": verbs[2],
    "construction": "caused-motion"
  })
  sentences.append({
    "sentence": f"{names[15]} {verbs[3]} the {complements[3][2][1]} {preposition_phrases[3]}.",
    "verb": verbs[3],
    "construction": "caused-motion"
  })
  
  # Resultative
  sentences.append({
    "sentence": f"{names[16]} {verbs[0]} the {complements[0][3][1]} {complements[0][3][2]}.",
    "verb": verbs[0],
    "construction": "resultative"
  })
  sentences.append({
    "sentence": f"{names[17]} {verbs[1]} the {complements[1][3][1]} {complements[1][3][2]}.",
    "verb": verbs[1],
    "construction": "resultative"
  })
  sentences.append({
    "sentence": f"{names[18]} {verbs[2]} the {complements[2][3][1]} {complements[2][3][2]}.",
    "verb": verbs[2],
    "construction": "resultative"
  })
  sentences.append({
    "sentence": f"{names[19]} {verbs[3]} the {complements[3][3][1]} {complements[3][3][2]}.",
    "verb": verbs[3],
    "construction": "resultative"
  })
  
  return pd.DataFrame(sentences)

gen_stimuli()

