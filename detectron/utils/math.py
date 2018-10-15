import numpy as np
from scipy import spatial


def cosine_similarity(A, B):
	if len(A.shape) == 1 and len(B.shape) == 1:
	    return 1 - spatial.distance.cosine(A, B)
	elif len(A.shape) == 2 and B.shape == A.shape:
		return np.array([cosine_similarity(A[i], B[i]) for i in xrange(len(A))])
	else:
		raise ValueError
