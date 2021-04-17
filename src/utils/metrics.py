import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import adjusted_mutual_info_score


def accuracy(true_labels, cluster_labels):
    cluster_labels_set = list(set(cluster_labels))
    true_labels_set = list(set(true_labels))
    costs = np.zeros([len(cluster_labels_set), len(true_labels_set)], dtype=float)

    for i in range(costs.shape[0]):
        c_label = cluster_labels_set[i]
        for j in range(costs.shape[1]):
            t_label = true_labels_set[j]
            for doc_id, doc_c_label in enumerate(cluster_labels):
                if doc_c_label == c_label and true_labels[doc_id] != t_label:
                    costs[i, j] += 1

    row_ind, col_ind = linear_sum_assignment(costs)  # assigning labels using Hungarian algorithm

    total_cost = costs[row_ind, col_ind].sum()
    total_accuracy = 1. - total_cost / len(true_labels)

    return np.round(total_accuracy, 5)


def adjusted_mutual_info(true_labels, cluster_labels):
    return np.round(adjusted_mutual_info_score(true_labels, cluster_labels), 5)
