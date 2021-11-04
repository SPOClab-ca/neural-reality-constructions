"""
Run Johnson & Goldberg (Jabberwocky construction) experiments.
"""
import argparse
import numpy as np
import pandas as pd
import pickle

import src.sent_encoder


parser = argparse.ArgumentParser()

# high-freq or low-freq
parser.add_argument('--condition', default='high-freq')

def print_log(*s):
  with open('experiments.log', 'a') as outf:
    print(*s)
    print(*s, file=outf)

args = parser.parse_args()
print_log(args)


template_data = pd.read_csv("data/jabberwocky-templated.csv")
with open("data/bnc.pkl", "rb") as f:
  bnc_data = pickle.load(f)


LAYER = 11
enc = src.sent_encoder.SentEncoder()


CONSTRUCTIONS = ['ditransitive', 'resultative', 'caused-motion', 'removal']

if args.condition == 'high-freq':
  PROTOTYPE_VERBS = ['gave', 'made', 'put', 'took']
elif args.condition == 'low-freq':
  PROTOTYPE_VERBS = ['handed', 'turned', 'placed', 'removed']
else:
  assert(False)


# Compute prototype vectors for each verb
prototype_vecs = {}
for proto_verb in PROTOTYPE_VERBS:
  prototype_vecs[proto_verb] = enc.avg_contextual_word_vec(bnc_data, proto_verb)[LAYER]


def is_congruent(cxn, verb):
  return CONSTRUCTIONS.index(cxn) == PROTOTYPE_VERBS.index(verb)


verb_dist_results = []

for cxn_type in CONSTRUCTIONS:
  cxn_sentences = template_data[template_data.construction == cxn_type].sentence.tolist()
  cxn_verbs = template_data[template_data.construction == cxn_type].nonse_verb.tolist()
  cxn_verb_vecs = enc.sentence_vecs(cxn_sentences, cxn_verbs)[:, LAYER]
  
  for proto_verb, proto_verb_vec in prototype_vecs.items():
    for i, cxn_verb_vec in enumerate(cxn_verb_vecs):
      dist = np.linalg.norm(proto_verb_vec - cxn_verb_vec)
      verb_dist_results.append(pd.Series({
        'cxn_sentence': cxn_sentences[i],
        'cxn': cxn_type,
        'verb': proto_verb,
        'congruent': is_congruent(cxn_type, proto_verb),
        'dist': dist,
      }))
      
verb_dist_results = pd.DataFrame(verb_dist_results)


# Summarize results
for verb in PROTOTYPE_VERBS:
  for cxn in CONSTRUCTIONS:
    m = verb_dist_results[(verb_dist_results.cxn == cxn) & (verb_dist_results.verb == verb)].dist.mean()
    sd = verb_dist_results[(verb_dist_results.cxn == cxn) & (verb_dist_results.verb == verb)].dist.std()
    print_log(cxn, verb, float(m), float(sd))

print_log('Mean congruent:', verb_dist_results[verb_dist_results.congruent].dist.mean())
print_log('Std congruent:', verb_dist_results[verb_dist_results.congruent].dist.std())
print_log('Mean incongruent:', verb_dist_results[~verb_dist_results.congruent].dist.mean())
print_log('Std incongruent:', verb_dist_results[~verb_dist_results.congruent].dist.std())
