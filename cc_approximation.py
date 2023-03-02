from itertools import combinations

"""
Approximation of Correlation Clustering
"""

def compute_all_clusterings(indices):
    """
    args:
        indices: indices of items
    """
    if len(indices)==1:
        yield [indices]
        return
    first = indices[0]
    for smaller in compute_all_clusterings(indices[1:]):
        # insert "first" in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n]+[[first]+subset]+smaller[n+1:]
        yield [[first]]+smaller


def compute_clustering_score(sims, clustering):
    """
    args:
        sims: similarity matrix
        clustering: list of lists denoting clusters
    """
    score = 0.
    for cluster in clustering:
        if len(cluster)>=2:
            combs = list(combinations(cluster, 2))
            for comb in combs:
                score += sims[comb]
    return score


def merge_2_clusters(current_clustering, indices):
    """
    merge 2 clusters of current clustering
    args:
        current_clustering: list of lists denoting clusters
        indices(tuple): indices of 2 clusters of current clustering
    """
    assert len(current_clustering)>1
    num_clusters = len(current_clustering)
    cluster1 = current_clustering[indices[0]]
    cluster2 = current_clustering[indices[1]]
    merged_cluster = cluster1+cluster2
    new_clustering = [merged_cluster]
    for i in range(num_clusters):
        if i!=indices[0] and i!=indices[1]:
            new_clustering.append(current_clustering[i])
    return new_clustering



def greedy_approximate_best_clustering(sims):
    """
    args:
        sims(numpy ndarray): similarity matrices, shape:[n_atoms, n_atoms]
        current_clustering: a list of lists denoting clusters
        current_score: current clustering score
    """
    num_atoms = sims.shape[0]
    current_cluster_indices = list(range(num_atoms))
    current_clustering = [[i] for i in current_cluster_indices]
    current_score = 0.
    merge_2_indices = list(combinations(current_cluster_indices, 2))
    best_clustering = current_clustering
    
    
    while(True):
        #merge 2 clusters hierachically
        
        #if len(current_clustering)==1: #cannot be merged anymore
        #    return current_clustering, current_score
        
        best_delta = 0
        for merge_index in merge_2_indices:
            new_clustering = merge_2_clusters(current_clustering, merge_index)
            new_score = compute_clustering_score(sims, new_clustering)
            delta = new_score-current_score
            if delta>best_delta:
                best_clustering = new_clustering
                best_delta = delta
                current_score = new_score
        if best_delta<=0:
            return best_clustering, current_score
        
        current_clustering = best_clustering
        current_num_clusters = len(current_clustering)
        if current_num_clusters==1:
            return current_clustering, current_score
        cluster_indices = list(range(current_num_clusters))
        merge_2_indices = list(combinations(cluster_indices, 2))







