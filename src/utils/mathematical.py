from numpy import inner
from numpy.linalg import norm


def cosine_similarity(a, b):
    na, nb = norm(a), norm(b)
    if na == .0 or nb == .0:
        return .0
    return inner(a, b)/(norm(a)*norm(b))
