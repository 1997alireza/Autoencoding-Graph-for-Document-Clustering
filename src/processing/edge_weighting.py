from src.utils.mathematical import cosine_similarity


def sentence_similarity_edge(embeddings_list_i, embeddings_list_j):
    """

    :param embeddings_list_i: sentences' embeddings of document i
    :param embeddings_list_j: sentences' embeddings of document j
    :return: mean pairwise cosine similarity, a float in range [-1, 1]
    """
    sum_pairwise_sim = .0

    for emb_i in embeddings_list_i:
        for emb_j in embeddings_list_j:
            sum_pairwise_sim += cosine_similarity(emb_i, emb_j)

    mean_pairwise_sim = sum_pairwise_sim / (len(embeddings_list_i) * len(embeddings_list_j))
    return mean_pairwise_sim
