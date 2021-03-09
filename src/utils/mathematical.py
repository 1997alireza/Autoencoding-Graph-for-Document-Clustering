from numpy import inner
from numpy.linalg import norm


def cosine_similarity(a, b):
    return inner(a, b)/(norm(a)*norm(b))
