import transformers
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import string
from gensim.models import KeyedVectors
from nltk import word_tokenize


BATCH_SIZE = 32
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class SentEncoder:
  def __init__(self, model_name='roberta-base'):
    self.model_name = model_name
    self.auto_model = None
    self.glove = None
    self.fasttext = None



  def contextual_token_vecs(self, sents):
    """Returns: (all_tokens, sentence_token_vecs) where:
    all_tokens is a List[List[tokens]], one list for each sentence.
    sentence_token_vecs is List[np.array(sentence length, 13, 768)], one array for each sentence.
    Ignore special tokens like [CLS] and [PAD].
    """
    if self.auto_model is None:
      self.auto_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
      self.auto_model = AutoModel.from_pretrained(self.model_name).to(device)
      self.pad_id = self.auto_tokenizer.pad_token_id

    all_tokens = []
    sentence_token_vecs = []

    for batch_ix in range(0, len(sents), BATCH_SIZE):
      batch_sentences = sents[batch_ix : batch_ix+BATCH_SIZE]

      ids = torch.tensor(self.auto_tokenizer(batch_sentences, padding=True)['input_ids']).to(device)

      with torch.no_grad():
        # (num_layers, batch_size, sent_length, 768)
        vecs = self.auto_model(
          ids,
          attention_mask=(ids != self.pad_id).float(),
          output_hidden_states=True)[2]
        vecs = np.array([v.detach().cpu().numpy() for v in vecs])

      for sent_ix in range(ids.shape[0]):
        tokens = []
        token_vecs = []

        for tok_ix in range(ids.shape[1]):
          if ids[sent_ix, tok_ix] not in self.auto_tokenizer.all_special_ids:
            cur_tok = self.auto_tokenizer.decode(int(ids[sent_ix, tok_ix]))
            # Exclude tokens that consist entirely of punctuation
            if cur_tok not in string.punctuation:
              tokens.append(cur_tok)
              token_vecs.append(vecs[:, sent_ix, tok_ix, :])

        all_tokens.append(tokens)
        sentence_token_vecs.append(np.array(token_vecs))

    return all_tokens, sentence_token_vecs


  def sentence_vecs(self, sents, verbs=None):
    """Returns List[np.array(13, 768)], one array for each sentence.
    If verbs are specified, return embedding at each verb. Otherwise, take the sum
    of all contextual vectors.
    """
    all_toks, all_vecs = self.contextual_token_vecs(sents)
    if verbs is None:
      return np.array([tok_vecs.sum(axis=0) for tok_vecs in all_vecs])
    else:
      rvecs = []
      for sent_toks, sent_vecs, main_verb in zip(all_toks, all_vecs, verbs):
        verb_ix = [ix for ix in range(len(sent_toks)) if sent_toks[ix].strip() in main_verb][0]
        rvecs.append(sent_vecs[verb_ix])
      return np.array(rvecs)


  def sent_vecs_from_word_vecs(self, sents, method):
    """Compute sentence vectors using traditional method of averaging word embeddings"""
    assert method == 'glove' or method == 'fasttext'

    if method == 'glove' and self.glove is None:
      self.glove = KeyedVectors.load_word2vec_format("./data/glove.840B.300d.txt", limit=200000)
    if method == 'fasttext' and self.fasttext is None:
      # No need to use the subword version (.bin) of fasttext because we can assume all of our words are
      # in-vocab, and this is faster.
      self.fasttext = KeyedVectors.load_word2vec_format("./data/crawl-300d-2M-subword.vec", limit=200000)

    results = []
    for sent in sents:
      wvecs = []
      for tok in word_tokenize(sent):
        if method == 'glove':
          wvecs.append(self.glove[tok])
        elif method == 'fasttext':
          wvecs.append(self.fasttext[tok])
        else:
          assert(False)
      results.append(np.array(wvecs).mean(axis=0))
    return np.array(results)[:, np.newaxis, :]


  def avg_contextual_word_vec(self, corpus, word):
    """Find average contextual word vector for a word, given a corpus (list of sentences,
    some of which contain the word).
    Returns: (num_layer, 768)
    """
    # Avoid picking up cases where our word is part of another word
    w_padded = " " + word + " "
    
    w_sentences = [s for s in corpus if w_padded in s]
    w_vecs = self.sentence_vecs(w_sentences, verbs=[word] * len(w_sentences))
    return w_vecs.mean(axis=0)
