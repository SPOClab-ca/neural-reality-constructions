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
from sentence_transformers import SentenceTransformer

import src.sent_encoder

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# In[2]:


df = pd.read_csv("../data/bencini-goldberg.csv")


# In[ ]:


enc = src.sent_encoder.SentEncoder()


# In[4]:


# Alternatively, use S-BERT. Make the sentence vectors have the same shape as our
# SentEncoder but as a 1-layer model.
sbert = SentenceTransformer('paraphrase-mpnet-base-v2')
def sbert_encode(sentences):
  sbert_vecs = sbert.encode(sentences)
  return [vecs[np.newaxis, :] for vecs in list(sbert_vecs)]


# In[5]:


sent_vecs = enc.sentence_vecs(df.sentence.tolist())
#sent_vecs = sbert_encode(df.sentence.tolist())

num_layers = sent_vecs[0].shape[0]


# ## PCA plot (dim=2)

# In[6]:


layer = 11
pca_model = sklearn.decomposition.PCA(n_components=2, whiten=True)
pca_vecs = pca_model.fit_transform(np.array(sent_vecs)[:, layer])


# In[7]:


pca_vecs.shape


# In[8]:


df['PC1'] = pca_vecs[:, 0]
df['PC2'] = pca_vecs[:, 1]


# In[9]:


sns.set(rc={'figure.figsize':(8, 8)})
sns.set(font_scale=0.7)
sns.set_style('white')
g = sns.scatterplot(x=df.PC1, y=df.PC2)
for _, row in df.iterrows():
  label_text = f"{row.sentence}\n{row.verb}, {row.construction}"
  g.text(row['PC1'] + 0.06, row['PC2'] - 0.05, label_text)
plt.show()


# In[10]:


matplotlib.rc_file_defaults()
sns.set(rc={'figure.figsize':(3, 3)})
sns.set_style('white')
sns.scatterplot(x=df.PC1, y=df.PC2, hue=df.verb)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()


# In[11]:


matplotlib.rc_file_defaults()
sns.set(rc={'figure.figsize':(3, 3)})
sns.set_style('white')
sns.scatterplot(x=df.PC1, y=df.PC2, hue=df.construction)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()


# ## Compute mean pairwise distances

# In[12]:


def verb_cxn_mean_distance(df, sent_vecs, layer):
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


# In[13]:


# Fisher = (avg between-class distance) / (avg within-class distance)
# https://sthalles.github.io/fisher-linear-discriminant/
# Higher = classes are more separable
def fisher_discriminant(clusters, sent_vecs, layer):
  centroid = np.array(sent_vecs)[:, layer].mean(axis=0)
  
  between_class_distances = []
  within_class_distances = []
  for cur_cluster in set(clusters):
    cluster_sent_vecs = []
    for i in range(len(clusters)):
      if clusters[i] == cur_cluster:
        cluster_sent_vecs.append(sent_vecs[i])
    cluster_centroid = np.array(cluster_sent_vecs)[:, layer].mean(axis=0)
    for cluster_sent_vec in cluster_sent_vecs:
      between_class_distances.append(np.linalg.norm(cluster_centroid - centroid))
      within_class_distances.append(np.linalg.norm(cluster_sent_vec[layer] - cluster_centroid))
    
  return np.mean(between_class_distances) / np.mean(within_class_distances)


def verb_cxn_fisher_discriminant(df, sent_vecs, layer):
  return fisher_discriminant(df.verb.tolist(), sent_vecs, layer), fisher_discriminant(df.construction.tolist(), sent_vecs, layer)


# Test case with 6 points. (Refactor into unit test later).
# Centroid total = (1, 0); Centroid a = (0, 0); Centroid b = (3, 0)
# Between-class distances = [1, 1, 1, 1, 2, 2]
# Within-class distances = [sqrt(2), sqrt(2), sqrt(2), sqrt(2), 1, 1]
# Answer should be 8 / (2 + 4 sqrt(2)) = 1.0448
fisher_discriminant(
  ['a', 'a', 'a', 'a', 'b', 'b'],
  np.array([
    [-1, 1], [1, 1], [1, -1], [-1, -1],
    [3, 1], [3, -1]]
  )[:, np.newaxis, :], 0)


# In[14]:


layer_results = []
for layer in range(num_layers):
  verb_fisher_discriminant, cxn_fisher_discriminant = verb_cxn_fisher_discriminant(df, sent_vecs, layer)
  layer_results.append({
    "layer": layer,
    "verb_fisher_discriminant": verb_fisher_discriminant,
    "cxn_fisher_discriminant": cxn_fisher_discriminant,
  })
layer_results = pd.DataFrame(layer_results)


# In[15]:


sns.set(rc={'figure.figsize':(4, 3)})
sns.set_style("white")
sns.lineplot(data=layer_results.melt(id_vars=["layer"]), x="layer", y="value", hue="variable")
plt.show()


# ## Repeat with lots of generated stimuli sets

# In[16]:


templated_df = pd.read_csv("../data/bencini-goldberg-templated.csv")


# In[17]:


results = []
for group in range(len(templated_df) // 16):
  df = templated_df[templated_df.group == group]
  sent_vecs = enc.sentence_vecs(df.sentence.tolist())
  #sent_vecs = sbert_encode(df.sentence.tolist())
  
  for layer in range(num_layers):
    verb_fisher_discriminant, cxn_fisher_discriminant = verb_cxn_fisher_discriminant(df, sent_vecs, layer)
    results.append({
      "group": group,
      "layer": layer,
      "verb_fisher_discriminant": verb_fisher_discriminant,
      "cxn_fisher_discriminant": cxn_fisher_discriminant,
    })

results = pd.DataFrame(results)


# In[18]:


sns.set_style("white")
sns.relplot(data=results.melt(id_vars=["group", "layer"]), x="layer", y="value", hue="variable",
            kind="line", ci="sd", aspect=4/3, height=3)
plt.show()

