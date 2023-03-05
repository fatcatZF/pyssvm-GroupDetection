import numpy as np 
from utils import create_edgeNode_relation
from features import compute_similarity
from cc_approximation import greedy_approximate_best_clustering
from structured_hinge_approximation import approximate_most_violated


class SSVM:
    def __init__(self, n_features=8):
        self.__init_weights(n_features)
        self.__init_l(n_features)

    def __init_weights(self, n_features):
        # initialize weights
        self.w = np.zeros(n_features)

    def __init_l(self, n_features):
        # initialize l
        self.l = 0.

    def fit_1_example_bcfw(self, example, ground, label, n_examples, wi, li, C, features):
        """
        train model with the BCFW algorithm
        args:
            example: the current example, shape: [n_atoms, n_timesteps, n_in]
            ground: the ground to compute heatmap, shape: [R,C,2]
            label: the label clustering, a list of lists
            n_examples: the number of examples
        """
        worst_clustering, hinge_loss, gmitre, delta_rep = approximate_most_violated(example, ground,
                                                                                    label, self.w, features)
        ws = (C/n_examples)*delta_rep
        ls = (C/n_examples)*gmitre
        gamma = (np.dot(wi-ws, self.w)+(C/n_examples)*(ls-li))/(((wi-ws)**2).sum()+1e-6)
        gamma = np.clip(gamma, 0, 1) #clip the value between 0 and 1
        wi_new = (1-gamma)*wi+gamma*ws 
        li_new = (1-gamma)*li+gamma*ls 
        self.w += wi_new-wi 
        self.l += li_new-li 
        return wi_new, li_new, gmitre, hinge_loss 
    

    def fit(self, examples, ground, labels, n_iters, C=10, verbose=0, 
            verbose_iters=100, pairwise_features=None):
        """
        train model with BCFW algorithm
        args:
            examples: a list of training examples 
            labels: a list of labels
            C: regularization parametre
        """
        n_features = self.w.shape[0]
        n_examples = len(examples)
        indices_training = np.arange(n_examples)

        wis = np.zeros((indices_training.shape[0], n_features))
        self.wis_dict = dict(zip(indices_training, wis))
        lis = np.zeros(len(indices_training))
        self.lis_dict = dict(zip(indices_training, lis))

        for i in range(n_iters):
            current_index = np.random.choice(indices_training, 1)[0]
            current_example = examples[current_index]
            current_label = labels[current_index]
            wi = self.wis_dict[current_index]
            li = self.lis_dict[current_index]

            if pairwise_features is not None:
                features = pairwise_features[current_index]
            else:
                features = None

            try:
                wi_new, li_new, gmitre, hinge_loss = self.fit_1_example_bcfw(current_example, ground,
                                                  current_label, n_examples, wi, li, C, features)
                self.wis_dict[current_index]=wi_new
                self.lis_dict[current_index]=li_new
                if verbose>0 and (i+1)%verbose_iters==0:
                    print("Iter: {:04d}".format(i+1),
                          "current example index: {:04d}".format(current_index),
                         "Group Mitre Loss: {:.10f}".format(gmitre),
                         "hinge Loss: {:.10f}".format(hinge_loss))
            
            except Exception as err:
                print("Exception: ", err)
                continue

        
    def predict(self, example, ground, features=None):
        """
        args:
          example(numpy ndarray, shape: [n_atoms, n_timesteps, n_in]
        return:
          predicted clustering: a list of lists denoting cluster
        """
        n_atoms = example.shape[0]
        rel_rec,rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
        if features is None:
            _, _ ,features = compute_similarity(example, ground)
        #compute similarity matrix
        sims_values = np.matmul(features, self.w)
        sims_values = np.diag(sims_values)
        sims = np.matmul(rel_send.T, np.matmul(sims_values, rel_rec))
        best_clustering, current_score = greedy_approximate_best_clustering(sims)
        return best_clustering, current_score
            


    

