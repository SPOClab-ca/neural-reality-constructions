import random
from gensim.models import KeyedVectors
import nltk


class WordVecPerturbation:
  """Methods to help perturb sentences by substituting nouns with closest word vectors"""
  
  def __init__(self):
    self.w2v = KeyedVectors.load_word2vec_format("./data/glove.840B.300d.txt", limit=50000)

    self.nouns_sg = set()
    self.nouns_pl = set()

    for w, pos in nltk.corpus.brown.tagged_words():
      if pos == 'NN':
        self.nouns_sg.add(w.lower())
      if pos == 'NNS':
        self.nouns_pl.add(w.lower())


  def capitalize(self, w):
    return w[0].upper() + w[1:]
  

  def closest_matching_words(self, w, topn=5):
    is_caps = w[0].isupper()
    w = w.lower()
    
    w2v_similar = self.w2v.most_similar(w, topn=topn*3)
    ans = []
    for sim_w, _ in w2v_similar:
      if w in self.nouns_sg and sim_w in self.nouns_sg or \
          w in self.nouns_pl and sim_w in self.nouns_pl:
        ans.append(self.capitalize(sim_w) if is_caps else sim_w)
    return ans[:topn]


  def perturb_sentence(self, sent, target_word, randomness=1):
    """Perturb target word of sentence. Randomly pick replacement word from top r closest vectors."""
    toks = sent.split(' ')
    assert target_word in toks

    if target_word.lower() not in self.w2v.vocab:
      return None, None

    ans = []
    replaced_word = None
    for tok in toks:
      if tok == target_word:
        choices = self.closest_matching_words(target_word, topn=randomness)
        if len(choices) == 0:
          return None, None

        ans.append(random.choice(choices))
        replaced_word = ans[-1]
      else:
        ans.append(tok)
    return ' '.join(ans), replaced_word
