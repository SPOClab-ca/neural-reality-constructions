"""
Run either Bencini & Goldberg (verb vs cxn) or Perek (cxn vs alternation) grouping
experiments on different sentence embeddings.
"""
import argparse
import numpy as np
import pandas as pd
import random; random.seed(12345)
from sentence_transformers import SentenceTransformer

import src.sent_encoder
from src.fisher_metric import fisher_discriminant


parser = argparse.ArgumentParser()

# bencini-goldberg or perek
parser.add_argument('--experiment', type=str, default='bencini-goldberg')

# roberta-base, bert-base-uncased, glove, fasttext, etc
parser.add_argument('--model_name', type=str, default='roberta-base')

# original or templates
parser.add_argument('--stimuli', type=str, default='original')

# average or verb (only for LMs)
parser.add_argument('--aggregate', type=str, default='average')


def print_log(*s):
  with open('experiments.log', 'a') as outf:
    print(*s)
    print(*s, file=outf)

args = parser.parse_args()
print_log(args)


# Load original stimuli or templates
if args.experiment == 'bencini-goldberg' and args.stimuli == 'original':
  stimuli = pd.read_csv("data/bencini-goldberg.csv")
elif args.experiment == 'bencini-goldberg' and args.stimuli == 'templates':
  stimuli = pd.read_csv("data/bencini-goldberg-templated.csv")
elif args.experiment == 'perek' and args.stimuli == 'original':
  stimuli = pd.read_csv("data/perek2012.csv")
elif args.experiment == 'perek' and args.stimuli == 'templates':
  stimuli = pd.read_csv("data/perek-templated.csv")
else:
  assert(False)


# If using SBERT: make the sentence vectors have the same shape as our
# SentEncoder but as a 1-layer model.
is_using_sbert = args.model_name.startswith('sbert-')
if is_using_sbert:
  sbert = SentenceTransformer(args.model_name[6:])
  def sbert_encode(sentences):
    sbert_vecs = sbert.encode(sentences)
    return np.array([vecs[np.newaxis, :] for vecs in list(sbert_vecs)])
else:
  enc = src.sent_encoder.SentEncoder(model_name=args.model_name)


# Encode sentences based on configuration specified
def encode_sentences(sentences):
  if args.model_name in ['glove', 'fasttext']:
    return enc.avg_word_vecs(sentences, method=args.model_name)
  elif is_using_sbert:
    return sbert_encode(sentences)
  elif args.aggregate == 'average':
    return enc.sentence_vecs(sentences)
  elif args.aggregate == 'verb':
    return enc.sentence_vecs(sentences, verbs=df.verb.tolist())
  else:
    assert(False)


def run_bencini_goldberg():
  # Compute sentence vectors
  results = []
  for group in range(len(stimuli) // 16):
    df = stimuli[stimuli.group == group]
    sent_vecs = encode_sentences(df.sentence.tolist())

    num_layers = sent_vecs.shape[1]
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

  # Output layer, fisher discriminants, standard deviation
  for layer in range(num_layers):
    print_log(
      layer,
      results[results.layer == layer].verb_fisher_discriminant.mean(),
      results[results.layer == layer].cxn_fisher_discriminant.mean(),
      results[results.layer == layer].null_fisher_discriminant.mean(),
      results[results.layer == layer].verb_fisher_discriminant.std(),
      results[results.layer == layer].cxn_fisher_discriminant.std(),
      results[results.layer == layer].null_fisher_discriminant.std()
    )


def run_perek():
  results = []
  for group in range(len(stimuli) // 16):
    df = stimuli[stimuli.group == group].reset_index(drop=True)
    sent_vecs = encode_sentences(df.sentence.tolist())

    num_layers = sent_vecs.shape[1]
    cxns = df.construction.unique()
    for layer in range(num_layers):
      for i1 in range(4):
        for i2 in range(i1+1, 4):
          cxn1 = cxns[i1]
          cxn2 = cxns[i2]
          cxn1_vecs = sent_vecs[df[df.construction == cxn1].index.tolist(), layer]
          cxn2_vecs = sent_vecs[df[df.construction == cxn2].index.tolist(), layer]

          pair_name = f"{cxn1}_vs_{cxn2}"
          fisher_score = fisher_discriminant(['a'] * 4 + ['b'] * 4, np.vstack([cxn1_vecs, cxn2_vecs]))
          results.append(pd.Series({
            "group": group,
            "layer": layer,
            "cxn1": cxn1,
            "cxn2": cxn2,
            pair_name: fisher_score,
          }))

  results = pd.DataFrame(results)

  # Todo: this logic is broken
  import pdb; pdb.set_trace()

  for layer in range(num_layers):
    print_log(
      layer
      #results[results.layer == layer].verb_fisher_discriminant.mean(),
      #results[results.layer == layer].verb_fisher_discriminant.std(),
    )


if args.experiment == 'bencini-goldberg':
  run_bencini_goldberg()
elif args.experiment == 'perek':
  run_perek()
else:
  assert(False)
