"""
Run Bencini and Goldberg verb vs cxn grouping experiment on different sentence embeddings.
"""
import argparse
import numpy as np
import pandas as pd

import src.sent_encoder
from src.fisher_metric import fisher_discriminant


parser = argparse.ArgumentParser()

# roberta-base, bert-base-uncased
parser.add_argument('--model_name', type=str, default='roberta-base')

# original or templates
parser.add_argument('--stimuli', type=str, default='original')

# average or verb (only for LMs)
# parser.add_argument('--aggregate', type=str, default='average')


def print_log(*s):
  with open('experiments.log', 'a') as outf:
    print(*s)
    print(*s, file=outf)

args = parser.parse_args()
print_log(args)


# Load original stimuli or templates
if args.stimuli == 'original':
  stimuli = pd.read_csv("data/bencini-goldberg.csv")
elif args.stimuli == 'templates':
  stimuli = pd.read_csv("data/bencini-goldberg-templated.csv")
else:
  assert(False)


# Compute sentence vectors
enc = src.sent_encoder.SentEncoder(model_name=args.model_name)
sent_vecs = enc.sentence_vecs(stimuli.sentence.tolist())
num_layers = sent_vecs.shape[1]


results = []
for group in range(len(stimuli) // 16):
  df = stimuli[stimuli.group == group]
  sent_vecs = enc.sentence_vecs(df.sentence.tolist())
  
  for layer in range(num_layers):
    verb_fisher_discriminant = fisher_discriminant(df.verb.tolist(), sent_vecs[:, layer])
    cxn_fisher_discriminant = fisher_discriminant(df.construction.tolist(), sent_vecs[:, layer])
    results.append({
      "group": group,
      "layer": layer,
      "verb_fisher_discriminant": verb_fisher_discriminant,
      "cxn_fisher_discriminant": cxn_fisher_discriminant,
    })
results = pd.DataFrame(results)


# Output layer, fisher discriminants, standard deviation
for layer in range(num_layers):
  print_log(
    layer,
    results[results.layer == layer].verb_fisher_discriminant.mean(),
    results[results.layer == layer].cxn_fisher_discriminant.mean(),
    results[results.layer == layer].verb_fisher_discriminant.std(),
    results[results.layer == layer].cxn_fisher_discriminant.std()
  )
