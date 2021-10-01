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


  def test_postprocess_standardize(self):
    sent_vecs = self.encoder.postprocess_standardize(self.encoder.sentence_vecs(self.sents))
    assert sent_vecs.shape == (2, 13, 768)
    assert self.encoder.corpus_means.shape == (13, 768)
    assert self.encoder.corpus_stds.shape == (13, 768)


  def test_sentence_vecs_with_verb(self):
    sent_vecs = self.encoder.sentence_vecs(self.sents, ['morning', 'are'])

    assert sent_vecs.shape == (2, 13, 768)
    assert np.array_equal(sent_vecs[0], self.all_vecs[0][1])
    assert np.array_equal(sent_vecs[1], self.all_vecs[1][1])


  def test_sent_vecs_from_word_vecs(self):
    sent_vecs = self.encoder.sent_vecs_from_word_vecs(self.sents, method='glove')
    assert sent_vecs.shape == (2, 1, 300)
    assert np.array_equal(
      sent_vecs[0][0],
      0.5*self.encoder.glove['Good'] + 0.5*self.encoder.glove['morning']
    )

    sent_vecs = self.encoder.sent_vecs_from_word_vecs(self.sents, method='fasttext')
    assert sent_vecs.shape == (2, 1, 300)
    assert np.array_equal(
      sent_vecs[0][0],
      0.5*self.encoder.fasttext['Good'] + 0.5*self.encoder.fasttext['morning']
    )


  def test_avg_contextual_word_vec(self):
    are_vec = self.encoder.avg_contextual_word_vec(self.sents, "are")

    assert are_vec.shape == (13, 768)
    assert np.array_equal(are_vec, self.all_vecs[1][1])
