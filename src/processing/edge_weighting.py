from src.utils.mathematical import cosine_similarity


def sentence_similarity_edge(sentences_idx_i, sentences_idx_j, embeddings):
    """

    :param sentences_idx_i:
    :param sentences_idx_j:
    :param embeddings:
    :return: mean pairwise cosine similarity, a float in range [-1, 1]
    """
    sum_pairwise_sim = .0
    for sent_idx_i in sentences_idx_i:
        for sent_idx_j in sentences_idx_j:
            sum_pairwise_sim += cosine_similarity(embeddings[sent_idx_i], embeddings[sent_idx_j])

    mean_pairwise_sim = sum_pairwise_sim / (len(sentences_idx_i) * len(sentences_idx_j))
    return mean_pairwise_sim
