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


  def test_perturb_sentence(self):
    new_sent, new_word = self.perturb.perturb_sentence("Pigs are eaten for food", "Pigs")
    assert new_sent == "Chickens are eaten for food"
    assert new_word == "Chickens"
