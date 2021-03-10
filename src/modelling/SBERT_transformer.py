from sentence_transformers import SentenceTransformer


def get_sentence_transformer():
    return SentenceTransformer('paraphrase-distilroberta-base-v1')
