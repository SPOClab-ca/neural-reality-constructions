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
from src.fisher_metric import fisher_discriminant

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


# Alternatively, use S-BERT. Make the sentence vectors have the same shape as our
# SentEncoder but as a 1-layer model.
sbert = SentenceTransformer('paraphrase-mpnet-base-v2')
def sbert_encode(sentences):
  sbert_vecs = sbert.encode(sentences)
  return [vecs[np.newaxis, :] for vecs in list(sbert_vecs)]


# In[ ]:


sent_vecs = enc.sentence_vecs(df.sentence.tolist())
#sent_vecs = sbert_encode(df.sentence.tolist())

num_layers = sent_vecs.shape[1]


# ## PCA plot (dim=2)

# In[6]:


layer = 11
pca_model = sklearn.decomposition.PCA(n_components=2, whiten=True)
pca_vecs = pca_model.fit_transform(sent_vecs[:, layer])


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


layer_results = []
for layer in range(num_layers):
  verb_fisher_discriminant = fisher_discriminant(df.verb.tolist(), sent_vecs[:, layer])
  cxn_fisher_discriminant = fisher_discriminant(df.construction.tolist(), sent_vecs[:, layer])
  layer_results.append({
    "layer": layer,
    "verb_fisher_discriminant": verb_fisher_discriminant,
    "cxn_fisher_discriminant": cxn_fisher_discriminant,
  })
layer_results = pd.DataFrame(layer_results)


# In[13]:


sns.set(rc={'figure.figsize':(4, 3)})
sns.set_style("white")
sns.lineplot(data=layer_results.melt(id_vars=["layer"]), x="layer", y="value", hue="variable")
plt.show()


# ## Repeat with lots of generated stimuli sets

# In[14]:


templated_df = pd.read_csv("../data/bencini-goldberg-templated.csv")


# In[15]:


results = []
for group in range(len(templated_df) // 16):
  df = templated_df[templated_df.group == group]
  sent_vecs = enc.sentence_vecs(df.sentence.tolist())
  #sent_vecs = sbert_encode(df.sentence.tolist())
  
  random_grouping = random.sample(['a', 'b', 'c', 'd'] * 4, 16)
  for layer in range(num_layers):
    verb_fisher_discriminant = fisher_discriminant(df.verb.tolist(), sent_vecs[:, layer])
    cxn_fisher_discriminant = fisher_discriminant(df.construction.tolist(), sent_vecs[:, layer])
    null_fisher_discriminant = fisher_discriminant(random_grouping, sent_vecs[:, layer])
    
    results.append({
      "group": group,
      "layer": layer,
      "verb_fisher_discriminant": verb_fisher_discriminant,
      "cxn_fisher_discriminant": cxn_fisher_discriminant,
      "null_fisher_discriminant": null_fisher_discriminant,
    })

results = pd.DataFrame(results)


# In[16]:


sns.set_style("white")
sns.relplot(data=results.melt(id_vars=["group", "layer"]), x="layer", y="value", hue="variable",
            kind="line", ci="sd", aspect=4/3, height=3)
plt.show()

