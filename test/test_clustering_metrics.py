import numpy as np
from src.clustering_metrics import distance_to_clustering
from src.clustering_metrics import fisher_discriminant


def test_distance_to_clustering():
  # Just needs a re-naming, no re-assignments. One item per class. 
  ypred = ['a', 'b', 'd', 'c', 'a']
  ytrue = ['a', 'b', 'c', 'd', 'a']
  dist = distance_to_clustering(ypred, ytrue)
  assert dist == 0

  # [[1 0 0]
  #  [1 1 0]
  #  [0 0 1]]  <- contingency_matrix
  # There is one off-the-diagonal element. distance = 1
  ypred = ['a', 'b', 'b', 'c']
  ytrue = ['a', 'a', 'b', 'c']
  dist = distance_to_clustering(ypred, ytrue)
  assert dist == 1

  # [[1 0 0]
  #  [1 1 1]
  #  [0 0 1]]  <- contingency_matrix
  # There are two off-the-diagonal elements. distance = 2
  ypred = ['a', 'b', 'b', 'b', 'c']
  ytrue = ['a', 'a', 'b', 'c', 'c']
  dist = distance_to_clustering(ypred, ytrue)
  assert dist == 2

  # [[0 0 1]
  #  [1 1 1]
  #  [1 0 0]]  <- contingency_matrix
  # There are two off-the-diagonal elements. distance = 2
  ypred = ['c', 'b', 'b', 'b', 'a']
  ytrue = ['a', 'a', 'b', 'c', 'c']
  dist = distance_to_clustering(ypred, ytrue)
  assert dist == 2

  # [[1 1 0]
  #  [0 1 1]
  #  [0 0 1]]  <- contingency_matrix
  # There are two off-the-diagonal elements. distance = 2
  ypred = ['a', 'a', 'b', 'b', 'c']
  ytrue = ['a', 'b', 'b', 'c', 'c']
  dist = distance_to_clustering(ypred, ytrue)
  assert dist == 2

  # Distance from pure-verb to pure-cxn clustering should be 12
  ypred = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'd', 'd', 'd', 'd']
  ytrue = ['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd']
  dist = distance_to_clustering(ypred, ytrue)
  assert dist == 12


def test_fisher_discriminant():
  # Test case with 6 points.
  # Centroid total = (1, 0); Centroid a = (0, 0); Centroid b = (3, 0)
  # Between-class distances = [1, 1, 1, 1, 2, 2]
  # Within-class distances = [sqrt(2), sqrt(2), sqrt(2), sqrt(2), 1, 1]
  # Answer should be 8 / (2 + 4 sqrt(2)) = 1.0448
  val = fisher_discriminant(
    ['a', 'a', 'a', 'a', 'b', 'b'],
    np.array([
      [-1, 1], [1, 1], [1, -1], [-1, -1],
      [3, 1], [3, -1]]
    ))

  assert val == 8 / (2 + 4 * 2**0.5)
