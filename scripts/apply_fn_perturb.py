"""
Script to replace sentences (short_fn_exemplars.csv), given target words, by swapping
target words with randomly selected nearest words in word2vec space.
"""
import pandas as pd
import json
import random

import src.wordvec_perturbation

random.seed(12345)


perturb_model = src.wordvec_perturbation.WordVecPerturbation()
df = pd.read_csv("./notebooks/short_fn_exemplars.csv")

out_df = []
for _, row in df.iterrows():
  noun_targets = json.loads(row.noun_targets)
  annotations = json.loads(row.annotations)

  for target_word in noun_targets:
    replaced_sent, replaced_word = perturb_model.perturb_sentence(
      row.text, target_word, randomness=50
    )
    if replaced_word is None:
      continue

    frame_element = None
    for fe_name, fe_text in annotations:
      if target_word in fe_text.split():
        frame_element = fe_name

    out_df.append({
      'frame': row.frame,
      'original_sent': row.text,
      'replaced_sent': replaced_sent,
      'original_word': target_word,
      'replaced_word': replaced_word,
      'frame_element': frame_element
    })

out_df = pd.DataFrame(out_df)
out_df.to_csv("short_fn_perturbed.csv", index=False)

print(f"Done, generated {len(out_df)} sentence pairs.")
