import numpy as np
from itertools import combinations
from features import compute_similarity
from group_mitre import compute_gmitre_loss
from cc_approximation import merge_2_clusters

def compute_features_matrix(features, n_atoms):
    """
    compute feature matrix given edge features
    args:
      features: [n_edges, n_in]
      n_atoms: number of atoms
    """
    assert features.shape[0]==n_atoms*(n_atoms-1)
    features_matrix = features.reshape(n_atoms, n_atoms-1, -1)
    features_matrix = features_matrix.tolist()
    for i in range(len(features_matrix)):
        feature = [0]*features.shape[1]
        features_matrix[i].insert(i, feature)
    features_matrix = np.array(features_matrix)
    
    return features_matrix


def compute_combined_rep(example, ground ,clustering, features=None):
    """
    compute combined feature representation
    args:
        example: [n_atoms, n_timesteps, n_shape]
        ground: [R, C, 2]
        clustering: a list of lists denoting clusters
    return:
        combined feature representation
    """
    n_atoms = example.shape[0]
    if features is None:
        _, _ ,features = compute_similarity(example, ground)
        #features shape: [n_atoms*(n_atoms-1), n_features]
    features_matrix = compute_features_matrix(features, n_atoms)
    
    combined_rep = np.zeros(features.shape[1])
    for cluster in clustering:
        combs = list(combinations(cluster, 2))
        for index in combs:
            combined_rep += features_matrix[index]
            
    return combined_rep


def compute_delta_rep(example, ground, label, predicted, features=None):
    """
    compute delta representation
    args:
      label: label clustering
      predicted: predicted clustering
      features: [n_edges, n_in]
    """
    combined_rep_label = compute_combined_rep(example, ground, label, features)
    combined_rep_pred = compute_combined_rep(example, ground, predicted, features)
    delta_rep = combined_rep_label-combined_rep_pred
    return delta_rep


def compute_structured_hinge(example, ground, label, predicted, w, features=None):
    """
    compute structured hinge loss
    args:
        example, shape: [n_atoms, n_timesteps, n_in]
        label/predicted: label and predicted clustering
    """
    gmitre = compute_gmitre_loss(label, predicted)
    delta_rep = compute_delta_rep(example, ground, label, predicted, features)
    return gmitre-np.dot(w, delta_rep), gmitre, delta_rep



def approximate_most_violated(example, ground, label, w, features=None):
    """
    greedy approximate most violated clustering
    args:
        example, shape: [n_atoms, n_timesteps, n_in]
        label: clustering, a list of lists denoting clusters
        w: weights
    """
    n_atoms = example.shape[0]
    current_cluster_indices = list(range(n_atoms))
    current_clustering = [[i] for i in current_cluster_indices]
    hinge_loss, gmitre, delta_rep = compute_structured_hinge(example, ground, label, current_clustering, w, features)
    merge_2_indices = list(combinations(current_cluster_indices, 2))
    worst_clustering = current_clustering
    
    while(True):
        #merge 2 clusters hierachically
        #if len(current_clustering)==1:
        #    return current_clustering, hinge_loss, gmitre, delta_rep
        most_delta = 0
        for merge_index in merge_2_indices:
            new_clustering = merge_2_clusters(current_clustering, merge_index)
            new_hinge_loss, new_gmitre, new_delta_rep = compute_structured_hinge(example, ground, 
                                                                                 label, new_clustering, w, features)
            delta = new_hinge_loss-hinge_loss
            if delta>most_delta:
                worst_clustering=new_clustering
                most_delta = delta
                hinge_loss = new_hinge_loss
                gmitre = new_gmitre
                delta_rep = new_delta_rep
        if most_delta==0:
            return worst_clustering, hinge_loss, gmitre, delta_rep
        
        current_clustering = worst_clustering
        current_num_clusters = len(current_clustering)
        if current_num_clusters==1:
            return current_clustering, hinge_loss, gmitre, delta_rep
        cluster_indices = list(range(current_num_clusters))
        merge_2_indices = list(combinations(cluster_indices, 2))

