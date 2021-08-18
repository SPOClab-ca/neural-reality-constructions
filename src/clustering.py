from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment


def distance_to_clustering(ypred, ytrue, verbose=False):
    cm = contingency_matrix(ypred, ytrue)
    row_ind, col_ind = linear_sum_assignment(-cm)
    if verbose:
        print(cm)
        print(row_ind, col_ind)
    return cm.sum() - cm[row_ind, col_ind].sum()