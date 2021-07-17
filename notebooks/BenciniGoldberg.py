#!/usr/bin/env python
# coding: utf-8

# # Run sentence embeddings on Bencini and Goldberg (2000) stimuli

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
import sklearn.decomposition

import src.sent_encoder

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# In[2]:


df = pd.read_csv("../data/bencini-goldberg.csv")


# In[3]:


enc = src.sent_encoder.SentEncoder()


# In[4]:


all_tokens, all_vecs = enc.contextual_token_vecs(df.sentence.tolist())


# In[5]:


sent_vecs = [tok_vecs.sum(axis=0) for tok_vecs in all_vecs]


# In[6]:


# Should be 16 (sentences) x 13 (layers) x 768
print(len(sent_vecs))
print(sent_vecs[0].shape)


# ## PCA plot (dim=2)

# In[7]:


layer = 11
pca_model = sklearn.decomposition.PCA(n_components=2)
pca_vecs = pca_model.fit_transform(np.array(sent_vecs)[:, layer])


# In[8]:


pca_vecs.shape


# In[9]:


df['PC1'] = pca_vecs[:, 0]
df['PC2'] = pca_vecs[:, 1]


# In[10]:


sns.set(rc={'figure.figsize':(8, 8)})
sns.set(font_scale=0.7)
sns.set_style('white')
g = sns.scatterplot(x=df.PC1, y=df.PC2)
for _, row in df.iterrows():
  label_text = f"{row.sentence}\n{row.verb}, {row.construction}"
  g.text(row['PC1'] + 2, row['PC2'] - 1, label_text)
plt.show()


# In[11]:


matplotlib.rc_file_defaults()
sns.set(rc={'figure.figsize':(3, 3)})
sns.set_style('white')
sns.scatterplot(x=df.PC1, y=df.PC2, hue=df.verb)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()


# In[12]:


matplotlib.rc_file_defaults()
sns.set(rc={'figure.figsize':(3, 3)})
sns.set_style('white')
sns.scatterplot(x=df.PC1, y=df.PC2, hue=df.construction)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()


# ## Compute mean pairwise distances

# In[13]:


def verb_cxn_mean_distance(layer):
  verb_distances = []
  cxn_distances = []
  for i in range(16):
    for j in range(i+1, 16):
      dist = np.linalg.norm(sent_vecs[i][layer] - sent_vecs[j][layer])
      if df.iloc[i].verb == df.iloc[j].verb:
        verb_distances.append(dist)
      if df.iloc[i].construction == df.iloc[j].construction:
        cxn_distances.append(dist)
  return np.mean(verb_distances), np.mean(cxn_distances)


# In[14]:


layer_results = []
for layer in range(13):
  verb_distance, cxn_distance = verb_cxn_mean_distance(layer)
  layer_results.append({
    "layer": layer,
    "verb_distance": verb_distance,
    "cxn_distance": cxn_distance,
  })
layer_results = pd.DataFrame(layer_results)


# In[15]:


sns.set(rc={'figure.figsize':(4, 3)})
sns.set_style("white")
sns.lineplot(data=layer_results.melt(id_vars=["layer"]), x="layer", y="value", hue="variable")
plt.show()

