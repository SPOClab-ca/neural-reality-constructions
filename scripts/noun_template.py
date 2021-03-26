"""
Script to generate some random sentences from templates by swapping nouns.
Outputs sentences to stdout.
"""
import numpy as np
import pandas as pd
import random

random.seed(12345)


# Setup singular and plural nouns
data = pd.read_csv("../data/vocabulary.csv")

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

singular_nouns = list(sorted(set(singular_nouns) - {np.nan}))
plural_nouns = list(sorted(set(plural_nouns) - {np.nan}))


def get_random_noun(sg_or_pl):
  if sg_or_pl == 'SINGLE':
    return random.choice(singular_nouns)
  elif sg_or_pl == 'PLURAL':
    return random.choice(plural_nouns)


def replace_nouns_in_sentence(sentence):
  while True:
    if 'SINGLE' in sentence:
      sentence = sentence.replace('SINGLE', get_random_noun('SINGLE'), 1)
    elif 'PLURAL' in sentence:
      sentence = sentence.replace('PLURAL', get_random_noun('PLURAL'), 1)
    else:
      break
  return sentence


# Generate from templates
templates_df = pd.read_csv("../data/hilpert.csv")
for template in templates_df.Template.tolist():
  for i in range(3):
    print(replace_nouns_in_sentence(template))
