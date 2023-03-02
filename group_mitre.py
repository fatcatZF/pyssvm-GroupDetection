
"""
Group Mitre
"""

def indices_to_clusters(l):
    """
    args:
        l: indices of clusters, e.g.. [0,0,1,1]
    return: clusters, e.g. [(0,1),(2,3)]
    """
    d = dict()
    for i,v in enumerate(l):
        d[v] = d.get(v,[])
        d[v].append(i)
    clusters = list(d.values())
    return clusters


def compute_mitre(target, predict):
    """
    args:
      target/predict: list of lists 
    """
    predict_lookup = {}
    for cluster in predict:
        for atom in cluster:
            predict_lookup[atom]=cluster
    total_misses = 0.
    total_corrects = 0.
    for cluster in target:
        size_cluster = len(cluster)
        total_corrects += size_cluster-1
        if size_cluster==1: # if the size of the cluster is 1, there are no missing links
            continue
        divided = [] #to record divided sets to compute missing links
        for atom in cluster:
            corr_cluster = predict_lookup[atom]
            if corr_cluster not in divided:
                divided.append(corr_cluster)
        total_misses+=len(divided)-1
    return (total_corrects-total_misses)/total_corrects


def create_counterPart(a):
    """
    add fake counter parts for each agent
    args:
      a: list of groups; e.g. a=[[0,1],[2],[3,4]]
    """
    a_p = []
    for group in a:
        if len(group)==1:#singleton
            element = group[0]
            element_counter = -(element+1)#assume element is non-negative
            new_group = [element, element_counter]
            a_p.append(new_group)
        else:
            a_p.append(group)
            for element in group:
                element_counter = -(element+1)
                a_p.append([element_counter])
    return a_p


def compute_groupMitre(target, predict):
    """
    compute group mitre
    args: 
      target,predict: list of groups; [[0,1],[2],[3,4]]
    return: recall, precision, F1
    """
    #create fake counter agents
    target_p = create_counterPart(target)
    predict_p = create_counterPart(predict)
    recall = compute_mitre(target_p, predict_p)
    precision = compute_mitre(predict_p, target_p)
    if recall==0 or precision==0:
        F1 = 0
    else:
        F1 = 2*recall*precision/(recall+precision)
    return recall, precision, F1


def compute_gmitre_loss(target, predict):
    _,_, F1 = compute_groupMitre(target, predict)
    return 1-F1

def compute_groupMitre_labels(target, predict):
    """
    compute group mitre given indices
    args: target, predict: list of indices of groups
       e.g. [0,0,1,1]
    """
    target = indices_to_clusters(target)
    predict = indices_to_clusters(predict)
    recall, precision, F1 = compute_groupMitre(target, predict)
    return recall, precision, F1







