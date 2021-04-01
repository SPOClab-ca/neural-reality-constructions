"""
Script to replace fixed words in sentences with hardcoded synonyms from a CSV file.
Outputs sentences to stdout.
"""
import numpy as np
import pandas as pd

synonym_df = pd.read_csv("../data/syn-replace.csv")
templates_df = pd.read_csv("../data/hilpert.csv")

for sent in templates_df.Sentence.tolist():
  sent1 = sent
  for _, row in synonym_df.iterrows():
    sent1 = sent1.replace(row.word, row.syn1)

  sent2 = sent
  for _, row in synonym_df.iterrows():
    sent2 = sent2.replace(row.word, row.syn2)

  print(sent1)
  print(sent2)
