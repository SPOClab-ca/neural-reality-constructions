import numpy as np
from src.clustering_metrics import fisher_discriminant


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
