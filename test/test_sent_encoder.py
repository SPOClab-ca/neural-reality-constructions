import unittest
import src.sent_encoder
import numpy as np

EPS = 1e-9

class TestSentEncoder(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.sents = ["Good morning", "You are drunk"]
    cls.encoder = src.sent_encoder.SentEncoder(model_name='roberta-base')

  def test_contextual_token_vecs(self):
    all_tokens, all_vecs = self.encoder.contextual_token_vecs(self.sents)

    assert len(all_vecs) == 2
    assert all_vecs[0].shape == (2, 13, 768)
    assert all_vecs[1].shape == (3, 13, 768)


  def test_sentence_vecs(self):
    sent_vecs = self.encoder.sentence_vecs(self.sents)

    assert len(sent_vecs) == 2
    assert sent_vecs[0].shape == (13, 768)
    assert sent_vecs[1].shape == (13, 768)
