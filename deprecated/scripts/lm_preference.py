"""
Given a dataframe of sentence pairs (text1, text2) that differ on one word, run an LM to calculate
which one it prefers. Store this result in a new column and write it to CSV.
"""
import argparse
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM


def get_masked_sequence(tokenizer, sent1, sent2):
  """Given a sentence pair that differ on exactly one token, return the token sequence
  with [MASK] token, index of mask token, and the two different tokens.
  """
  toks1 = tokenizer(sent1)['input_ids']
  toks2 = tokenizer(sent2)['input_ids']

  seq_with_mask = []
  masked_ix = None
  dtok1 = None
  dtok2 = None
  for ix in range(len(toks1)):
    if toks1[ix] != toks2[ix]:
      # Sequence is different here, so replace with [MASK].
      seq_with_mask.append(tokenizer.mask_token_id)
      masked_ix = ix
      dtok1 = toks1[ix]
      dtok2 = toks2[ix]
    else:
      seq_with_mask.append(toks1[ix])

  return seq_with_mask, masked_ix, dtok1, dtok2


# Return true if the list of tokens differs in exactly one place
def is_single_diff(toks1, toks2):
  if len(toks1) != len(toks2):
    return False
  
  diff_toks = 0
  for ix in range(len(toks1)):
    if toks1[ix] != toks2[ix]:
      diff_toks += 1
  
  return diff_toks == 1


def works_for_model(tokenizer, sent1, sent2):
  toks1 = tokenizer.tokenize(sent1)
  toks2 = tokenizer.tokenize(sent2)
  return is_single_diff(toks1, toks2)


def run_mlm(df, model_name='roberta-base'):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForMaskedLM.from_pretrained(model_name)

  # Make binary choice for a single sentence pair
  def mlm_sentence_pair(sent1, sent2):
    if not works_for_model(tokenizer, sent1, sent2):
      return None
    masked_toks, masked_ix, dtok1, dtok2 = get_masked_sequence(tokenizer, sent1, sent2)
    logit1 = model(torch.tensor([masked_toks])).logits[0, masked_ix, dtok1]
    logit2 = model(torch.tensor([masked_toks])).logits[0, masked_ix, dtok2]
    return float(logit2 - logit1)

  res = [mlm_sentence_pair(row.text1, row.text2) for _, row in df.iterrows()]
  df[model_name] = res
  return df


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', type=str)
  args = parser.parse_args()

  assert args.input.endswith(".csv")
  out_file_name = args.input[:-4] + "-out.csv"

  df = pd.read_csv(args.input)
  out_df = run_mlm(df)
  df.to_csv(out_file_name, index=False)
  print(f"Done, processed {len(df)} sentence pairs. Output: {out_file_name}")


main()
