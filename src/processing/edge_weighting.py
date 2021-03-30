from src.utils.mathematical import cosine_similarity


def sentence_similarity_edge(sentences_idx_i_tuple, sentences_idx_j_tuple, embeddings):
    """

    :param sentences_idx_i_tuple:
    :param sentences_idx_j_tuple:
    :param embeddings:
    :return: mean pairwise cosine similarity, a float in range [-1, 1]
    """
    sum_pairwise_sim = .0
    for sent_i_doc_idx, sent_i_sent_idx in sentences_idx_i_tuple:
        for sent_j_doc_idx, sent_j_sent_idx in sentences_idx_j_tuple:
            sum_pairwise_sim += cosine_similarity(
                embeddings[sent_i_doc_idx][sent_i_sent_idx],
                embeddings[sent_j_doc_idx][sent_j_sent_idx])

    mean_pairwise_sim = sum_pairwise_sim / (len(sentences_idx_i_tuple) * len(sentences_idx_j_tuple))
    return mean_pairwise_sim
