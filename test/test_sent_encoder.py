import unittest
import src.sent_encoder
import numpy as np

EPS = 1e-9

class TestSentEncoder(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.sents = ["Good morning", "You are drunk"]
    cls.encoder = src.sent_encoder.SentEncoder(model_name='roberta-base')
    cls.all_toks, cls.all_vecs = cls.encoder.contextual_token_vecs(cls.sents)

  def test_contextual_token_vecs(self):
    assert len(self.all_vecs) == 2
    assert self.all_vecs[0].shape == (2, 13, 768)
    assert self.all_vecs[1].shape == (3, 13, 768)


  def test_sentence_vecs(self):
    sent_vecs = self.encoder.sentence_vecs(self.sents)

    assert sent_vecs.shape == (2, 13, 768)


  def test_sentence_vecs_with_verb(self):
    sent_vecs = self.encoder.sentence_vecs(self.sents, ['morning', 'are'])

    assert sent_vecs.shape == (2, 13, 768)
    assert np.array_equal(sent_vecs[0], self.all_vecs[0][1])
    assert np.array_equal(sent_vecs[1], self.all_vecs[1][1])
