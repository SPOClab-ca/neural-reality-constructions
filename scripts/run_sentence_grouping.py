"""
Run either Bencini & Goldberg (sentence sorting) experiments on different
sentence embeddings.
"""
import argparse
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

import src.sent_encoder
from src.clustering_metrics import distance_to_clustering


parser = argparse.ArgumentParser()

# original/templates/german/italian/spanish
parser.add_argument('--dataset', type=str, default='templates')
parser.add_argument('--model_name', type=str, default='roberta-base')

def print_log(*s):
  with open('experiments.log', 'a') as outf:
    print(*s)
    print(*s, file=outf)

args = parser.parse_args()
print_log(args)


EXPERIMENT_DATA_FILES = {
  'original': "data/bencini-goldberg.csv",
  'templates': "data/bencini-goldberg-templated.csv",
  'german': "data/sorting-german.csv",
  'italian': "data/sorting-italian.csv",
  'spanish': "data/sorting-spanish.csv",
}
stimuli = pd.read_csv(EXPERIMENT_DATA_FILES[args.dataset])
enc = src.sent_encoder.SentEncoder(model_name=args.model_name)


def run_bencini_goldberg():
  # Compute sentence vectors
  results = []
  for group in range(len(stimuli) // 16):
    df = stimuli[stimuli.group == group]
    sent_vecs = enc.sentence_vecs(df.sentence.tolist())

    # Second-last layer
    layer = sent_vecs.shape[1] - 2

    cluster_alg = AgglomerativeClustering(n_clusters=4)
    cluster_labels = cluster_alg.fit_predict(sent_vecs[:, layer])

    dist_to_verb = distance_to_clustering(cluster_labels, df.verb.tolist())
    dist_to_cxn = distance_to_clustering(cluster_labels, df.construction.tolist())

    results.append({
      "group": group,
      "cluster4_distance_to_verb": dist_to_verb,
      "cluster4_distance_to_cxn": dist_to_cxn
    })

  results = pd.DataFrame(results)

  # Output layer, fisher discriminants, standard deviation
  print_log(
    results.cluster4_distance_to_verb.mean(),
    results.cluster4_distance_to_cxn.mean(),
    results.cluster4_distance_to_verb.std(),
    results.cluster4_distance_to_cxn.std()
  )


run_bencini_goldberg()
