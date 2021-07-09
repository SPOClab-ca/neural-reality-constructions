"""
Script to replace sentences (short_fn_exemplars.csv), given target words, by swapping
target words with randomly selected nearest words in word2vec space. Also generate annotator
spreadsheet by randomizing the order of sentences.
"""
import numpy as np
import pandas as pd
import json
import random

import src.wordvec_perturbation

INPUT_EXEMPLARS_CSV = "./notebooks/short_fn_exemplars.csv"
OUTPUT_PERTURBED_CSV = "short_fn_perturbed.csv"
OUTPUT_ANNOTATOR_CSV = "paired_for_annotator.csv"

random.seed(12345)
np.random.seed(12345)


perturb_model = src.wordvec_perturbation.WordVecPerturbation()
df = pd.read_csv(INPUT_EXEMPLARS_CSV)

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
out_df.to_csv(OUTPUT_PERTURBED_CSV, index=False)

print(f"Generated {len(out_df)} perturbed sentence pairs to {OUTPUT_PERTURBED_CSV}.")


# Randomize for annotation
annotator_df = out_df.sample(frac=1)
annotator_df['random'] = np.random.uniform()
annotator_df['text1'] = np.where(annotator_df.random < 0.5, annotator_df.original_sent, annotator_df.replaced_sent)
annotator_df['text2'] = np.where(annotator_df.random < 0.5, annotator_df.replaced_sent, annotator_df.original_sent)
annotator_df = annotator_df[['text1', 'text2']]
annotator_df.to_csv(OUTPUT_ANNOTATOR_CSV, index=False)

print(f"Generated {len(annotator_df)} annotation sentence pairs to {OUTPUT_ANNOTATOR_CSV}.")
