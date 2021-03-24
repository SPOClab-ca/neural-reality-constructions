#!/usr/bin/env python
# coding: utf-8

# # Explore Warstadt's vocabulary file

# In[1]:


import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import torch
from collections import defaultdict, Counter
import random
import math
import pickle

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# In[2]:


data = pd.read_csv("../data/vocabulary.csv")


# In[36]:


data.head(5)


# ## Examples of each CCG tag

# In[29]:


for pos, count in data.category.value_counts().items():
  if count >= 10:
    print(count, pos, data[data.category == pos].sample(3).expression.tolist())


# ## Extract singular and plural nouns

# In[52]:


df_nouns = data[data.category == 'N']
singular_nouns = []
plural_nouns = []

for _, row in data[(data.category == 'N') & (data.properNoun != 1)].iterrows():
  if row.pl == 1:
    singular_nouns.append(row.singularform)
    plural_nouns.append(row.expression)
  else:
    singular_nouns.append(row.expression)
    plural_nouns.append(row.pluralform)

singular_nouns = list(set(singular_nouns) - {np.nan})
plural_nouns = list(set(plural_nouns) - {np.nan})


# In[55]:


print(len(singular_nouns))
print(singular_nouns[:5])


# In[56]:


print(len(plural_nouns))
print(plural_nouns[:5])

