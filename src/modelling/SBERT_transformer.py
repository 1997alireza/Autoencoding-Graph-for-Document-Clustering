from sentence_transformers import SentenceTransformer


def get_sentence_transformer():
    return SentenceTransformer('bert-large-nli-stsb-mean-tokens')
