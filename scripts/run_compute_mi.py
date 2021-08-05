import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, "../")
from src.sent_encoder import SentEncoder

df = pd.read_csv("../notebooks/templated_stimuli.csv")

parser = argparse.ArgumentParser()
parser.add_argument("--modelname", type=str, default="nyu-mll/roberta-med-small-1M-1")
args = parser.parse_args()
print(args)

model_filename = args.modelname.replace("/", "_")
cached_fname = f"../notebooks-zining/templated_stimuli_{model_filename}.pkl"
if not os.path.exists(cached_fname):
    encoder = SentEncoder(args.modelname)
    res = np.array(encoder.sentence_vecs(df.sentence.tolist()))  # np.array(n_data, n_layer, n_model_dim)
    print("res.shape:", res.shape)
    torch.save(res, cached_fname)
    print ("Cached! Saved to", cached_fname)

all_encoded = torch.load(cached_fname)

###########################################
###   Scripts for plotting dendrograms  ###
###########################################

# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def make_dendrogram_plots(layer):
    encoded = all_encoded[:, layer, :]
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(encoded)
    
    fig = plt.figure(figsize=(12,8))
    plot_dendrogram(model, truncate_mode="level", p=3, orientation="top")
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.title("Hierarchical Clustering Dendrogram")
    plt.savefig(f"../notebooks-zining/0730_dendrogram/dendrogram_16_{model_filename}_layer{layer}.png")

    fig = plt.figure(figsize=(12,8))
    plot_dendrogram(model, truncate_mode="level", p=1, orientation="top")
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.title("Hierarchical Clustering Dendrogram")
    plt.savefig(f"../notebooks-zining/0730_dendrogram/dendrogram_4_{model_filename}_layer{layer}.png")
    
    plt.clf()


############################################
###   Scripts for quantitative analysis  ###
############################################
verbs = ['tore', 'punched', 'hit', 'pulled', 'threw', 'cut', 'got', 'sliced', 'kicked', 'pushed']
constructions = ["transitive", "ditransitive", "caused-motion", "resultative"]

def export_cluster_info_stats(df):
    report = {"LM": [], "n_clusters": [], "I(Y;V)": [], "I(Y;C)": []}
    report["LM"] += [model_filename] * 2
    mi_y_v_score = normalized_mutual_info_score(df[f"{model_filename}_cluster_outof4"], df["verb"])
    mi_y_c_score = normalized_mutual_info_score(df[f"{model_filename}_cluster_outof4"], df["construction"])
    report["n_clusters"].append(4)
    report["I(Y;V)"].append(mi_y_v_score)
    report["I(Y;C)"].append(mi_y_c_score)
    mi_y_v_score = normalized_mutual_info_score(df[f"{model_filename}_cluster_outof16"], df["verb"])
    mi_y_c_score = normalized_mutual_info_score(df[f"{model_filename}_cluster_outof16"], df["construction"])
    report["n_clusters"].append(16)
    report["I(Y;V)"].append(mi_y_v_score)
    report["I(Y;C)"].append(mi_y_c_score)
    return pd.DataFrame(report)
    

def export_cluster_raw_stats(df):
    report = {"LM": [], "n_clusters": [], "cluster_id": [], "n_in_cluster": [], 
              "by_construction": [], "construction_std": [],
              "by_verb": [], "verb_std": []}
    
    report["LM"] += [model_filename] * 20
    report["n_clusters"] += ([4 for i in range(4)]+[16 for i in range(16)])
    
    for cid in range(4):
        df_c = df[df[f"{model_filename}_cluster_outof4"] == cid]
        verb_ns = df_c.groupby(["verb"]).count()["sentence"]  # pd.Series
        verb_counts = np.array([verb_ns.get(v, 0) for v in verbs])
        verb_props = verb_counts / verb_counts.sum()
        verb_std = np.std(verb_props)
        construct_ns = df_c.groupby(["construction"]).count()["sentence"]
        construct_counts = np.array([construct_ns.get(c,0) for c in constructions])
        construct_props = construct_counts / construct_counts.sum()
        construct_std = np.std(construct_props)
        
        report["cluster_id"].append(cid)
        report["n_in_cluster"].append(len(df_c))
        report["by_verb"].append(verb_counts.tolist())
        report["verb_std"].append(verb_std)
        report["by_construction"].append(construct_counts.tolist())
        report["construction_std"].append(construct_std)
        
    for cid in range(16):
        df_c = df[df[f"{model_filename}_cluster_outof16"] == cid]
        verb_ns = df_c.groupby(["verb"]).count()["sentence"]  # pd.Series
        verb_counts = np.array([verb_ns.get(v, 0) for v in verbs])
        verb_props = verb_counts / verb_counts.sum()
        verb_std = np.std(verb_props)
        construct_ns = df_c.groupby(["construction"]).count()["sentence"]
        construct_counts = np.array([construct_ns.get(c,0) for c in constructions])
        construct_props = construct_counts / construct_counts.sum()
        construct_std = np.std(construct_props)
        
        report["cluster_id"].append(cid)
        report["n_in_cluster"].append(len(df_c))
        report["by_verb"].append(verb_counts.tolist())
        report["verb_std"].append(verb_std)
        report["by_construction"].append(construct_counts.tolist())
        report["construction_std"].append(construct_std)
    return pd.DataFrame(report)

all_raw_reports = []
all_info_reports = [] 
for layer in range(all_encoded.shape[1]):
    encoded = all_encoded[:, layer, :]  # (N, D)
    
    make_dendrogram_plots(layer)
    # Can I find which sentence goes into which branch?
    cluster = AgglomerativeClustering(n_clusters=4)
    labels = cluster.fit_predict(encoded)
    df[f"{model_filename}_cluster_outof4"] = labels   

    # Among each of these 4 groups, if we further split into 4, 
    # are there imbalanaces among constructions?
    cluster = AgglomerativeClustering(n_clusters=16)
    labels = cluster.fit_predict(encoded)
    df[f"{model_filename}_cluster_outof16"] = labels

    raw_report = export_cluster_raw_stats(df)
    raw_report["layer"] = [layer] * len(raw_report)
    info_report = export_cluster_info_stats(df)
    info_report["layer"] = [layer] * len(info_report)
    
    df.drop(columns=[f"{model_filename}_cluster_outof4", f"{model_filename}_cluster_outof16"])
    
    all_raw_reports.append(raw_report)
    all_info_reports.append(info_report)

all_layers_raw_report = pd.concat(all_raw_reports, axis=0)
all_layers_raw_report.to_csv(f"../notebooks-zining/0730_dendrogram/report_raw_{model_filename}.csv", index=False)
all_layers_info_report = pd.concat(all_info_reports, axis=0)
all_layers_info_report.to_csv(f"../notebooks-zining/0730_dendrogram/report_MI_{model_filename}.csv", index=False)
