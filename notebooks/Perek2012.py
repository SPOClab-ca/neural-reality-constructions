#!/usr/bin/env python
# coding: utf-8

# # Run sentence embeddings on Perek (2012) stimuli

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


df = pd.read_csv("../data/perek2012.csv")


# In[3]:


enc = src.sent_encoder.SentEncoder()


# In[ ]:


sent_vecs = enc.sentence_vecs(df.sentence.tolist())

num_layers = sent_vecs.shape[1]


# ## PCA plot (dim=2)

# In[5]:


layer = 11
pca_model = sklearn.decomposition.PCA(n_components=2, whiten=True)
pca_vecs = pca_model.fit_transform(sent_vecs[:, layer])


# In[6]:


pca_vecs.shape


# In[7]:


df['PC1'] = pca_vecs[:, 0]
df['PC2'] = pca_vecs[:, 1]


# In[8]:


sns.set(rc={'figure.figsize':(8, 8)})
sns.set(font_scale=0.7)
sns.set_style('white')
g = sns.scatterplot(x=df.PC1, y=df.PC2)
for _, row in df.iterrows():
  label_text = f"{row.sentence}\n{row.construction}"
  g.text(row['PC1'] + 0.06, row['PC2'] - 0.05, label_text)
plt.show()


# In[9]:


matplotlib.rc_file_defaults()
sns.set(rc={'figure.figsize':(3, 3)})
sns.set_style('white')
sns.scatterplot(x=df.PC1, y=df.PC2, hue=df.construction)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()

