import numpy as np
from src.clustering_metrics import distance_to_clustering


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
