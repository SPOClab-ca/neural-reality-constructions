import numpy as np


# Fisher = (avg between-class distance) / (avg within-class distance)
# https://sthalles.github.io/fisher-linear-discriminant/
# Higher = classes are more separable
def fisher_discriminant(clusters, sent_vecs):
  centroid = np.array(sent_vecs).mean(axis=0)
  
  between_class_distances = []
  within_class_distances = []
  for cur_cluster in set(clusters):
    cluster_sent_vecs = []
    for i in range(len(clusters)):
      if clusters[i] == cur_cluster:
        cluster_sent_vecs.append(sent_vecs[i])
    cluster_centroid = np.array(cluster_sent_vecs).mean(axis=0)
    for cluster_sent_vec in cluster_sent_vecs:
      between_class_distances.append(np.linalg.norm(cluster_centroid - centroid))
      within_class_distances.append(np.linalg.norm(cluster_sent_vec - cluster_centroid))
    
  return np.mean(between_class_distances) / np.mean(within_class_distances)
