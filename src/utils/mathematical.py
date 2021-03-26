from numpy import inner
from numpy.linalg import norm
import numpy as np


def cosine_similarity(a, b):
    return inner(a, b)/(norm(a)*norm(b))


def MSE(a, b):
    return np.sum((a-b)**2) / len(a)
