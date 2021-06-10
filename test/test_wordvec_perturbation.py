import unittest
import src.wordvec_perturbation

class TestWordVecPerturbation(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.perturb = src.wordvec_perturbation.WordVecPerturbation()


  def test_morph_dict_setup(self):
    assert 'women' in self.perturb.nouns_pl
    assert 'women' not in self.perturb.nouns_sg
    assert 'sheep' in self.perturb.nouns_sg
    assert 'sheep' in self.perturb.nouns_pl


  def test_word_perturb(self):
    assert 'cow' in self.perturb.closest_matching_words("pig")
    assert 'cows' in self.perturb.closest_matching_words("pigs")
    assert 'Cow' in self.perturb.closest_matching_words("Pig")
    assert 'Cows' in self.perturb.closest_matching_words("Pigs")
