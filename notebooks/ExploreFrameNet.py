#!/usr/bin/env python
# coding: utf-8

# # Explore FrameNet 1.7

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


# ## Look at some annotated sentences
# 
# 210k sentences, of which 200k are exemplars and 10k are full text.
# 
# Only the exemplars have frame annotations, so we'll only use the exemplars.

# In[2]:


print("Exemplars:", len(fn.exemplars()))
print("Full text:", len(fn.ft_sents()))
print("Total:", len(fn.sents()))


# In[3]:


fn_sents = fn.exemplars()


# In[4]:


fn_sents[12345]


# ## Distribution of sentence lengths
# 
# Average sentence is about 22 tokens, sharp dropoff at 40 for some reason.

# In[5]:


# Much faster than nltk.tokenize.word_tokenize
def approx_len(sent):
  return len(sent.split())


# In[6]:


counter = defaultdict(int)
for sent in fn_sents:
  ntok = approx_len(sent.text)
  counter[ntok] += 1


# In[7]:


plt.figure(figsize=(15, 5))
plt.bar(counter.keys(), counter.values())
plt.xlim((0, 120))
plt.show()


# ## Look at some shorter sentences

# In[8]:


short_sentences = [sent for sent in fn_sents[:200] if approx_len(sent.text) < 10]


# In[9]:


[sent.text for sent in short_sentences]


# In[10]:


len([sent for sent in fn_sents if approx_len(sent.text) <= 10])


# ## Get sentences of a specific frame
# 
# Here, there are 336 sentences labelled with the "Commerce_buy" frame.

# In[11]:


buying_frame = fn.frames("Commerce_buy")[0]


# In[12]:


buying_frame


# In[13]:


len(fn.exemplars(frame="Commerce_buy"))


# ## Get all frame elements of sentences
# 
# Easy to extract spans for frame elements, but they can be whole phrases, proper nouns, pronouns, etc.
# 
# Target seems to always be one word, in the set of words that lexically trigger the frame.

# In[14]:


example_sent = fn.exemplars(frame="Commerce_buy")[0]
example_text = example_sent.text


# In[15]:


example_sent


# In[16]:


print(example_sent.Target)
print(example_sent.FE[0])


# In[17]:


print("Target:", example_text[example_sent.Target[0][0]:example_sent.Target[0][1]])


# In[18]:


for span_start, span_end, fe_name in example_sent.FE[0]:
  print(f"{fe_name}: '{example_text[span_start:span_end]}'")


# ## Most popular frames
# 
# About 1k unique frames, most popular frame is Self_motion, with 6.4k sentences (3% of total).

# In[19]:


frame_count = defaultdict(int)
for sent in fn_sents:
  frame_count[sent.frame.name] += 1


# In[20]:


for frame_name, count in sorted(frame_count.items(), key=lambda kv: kv[1], reverse=True)[:15]:
  print(frame_name, count)


# In[21]:


len(frame_count)


# ## Look at frame-to-frame relations

# In[59]:


def frame_parents(frame_name):
  frame = fn.frames(frame_name)[0]
  ret = []
  for rel in frame.frameRelations:
    if rel.get("Parent") is not None and rel.Child.name == frame_name:
      ret.append(rel.Parent.name)
  return ret


# Commerce_buy frame inherits from Getting, and has 7 direct frame-to-frame relations.
# 
# This can be explored interactively: https://framenet.icsi.berkeley.edu/fndrupal/FrameGrapher.
# 
# Also has FE relations, eg, Getting.Recipient = Commerce_buy.Buyer.

# In[67]:


frame_parents("Commerce_buy")


# In[83]:


list(fn.frames("Commerce_buy")[0].frameRelations)


# In[81]:


list(fn.frames("Commerce_buy")[0].frameRelations[1].feRelations)

