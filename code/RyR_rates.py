import scipy
from scipy import linalg
import numpy as np

def getIC(A, eps=1e-15):
    u, s, vh = scipy.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    n = null_space.shape[0]
    assert n > 0, 'Matrix has full rank, empty null space.'
    for i in range(n):
	null_space[i,:] /= sum(null_space[i,:])
    if n>1:
	print "Model is not fully coupled, there are %d independent systems"%n
	null_space = sum(null_space)/n
    else:
	null_space = null_space[0]
    return null_space

state_names = ["RI", "R", "I", "O"]
ind = dict((state, ind) for ind, state in enumerate(state_names))
Ca = np.array([0.14, 0.30, 0.5, 10., 20.])
N = 4
Ni = 2
kim  = 0.01 # 1/ms
kom  = 2.0  # 1/ms

kd_i = 30.0 # um/ms
kd_o = 2.0  # um/ms^N (0.6-1.0)
ki   = (Ca/kd_i)**Ni
ko   = (Ca/kd_o)**N

A = np.zeros((len(state_names), len(state_names)))
for run in [0]:
    A[:] = 0
    A[ind["R"], ind["RI"]] = kim
    A[ind["O"], ind["I"]]  = kim
    A[ind["RI"],ind["I"]] = kom
    A[ind["R"], ind["O"]]  = kom
    A[ind["RI"],ind["R"]] = ki[run]
    A[ind["I"], ind["O"]]  = ki[run]
    A[ind["I"], ind["RI"]] = ko[run]
    A[ind["O"], ind["R"]]  = ko[run]
    for i in range(len(state_names)):
        A[i, i] = -A[:,i].sum()

    #A = A.transpose()
    IC = getIC(A)
    for state in state_names:
        print "{:2s}: {:.3f}".format(state, IC[ind[state]])
    print "N :", N

    print "                        ", "   ".join("{:.3f}".format(value) for value in Ca), "um"
    print "open wait time from R: ", "  ".join("{:.3g}".format(1./value) for value in ko), "ms"
    #print "                       {:.2f}   {:.2f}  {:.2f}  {:.2f} um".format(Ca[0], Ca[1], Ca[2], Ca[-1])
    #print "open wait time from R: {:.1f}  {:.1f}   {:.1f}   {:.1f} ms".format(1./(ko[0]), 1./(ko[1]), 1./(ko[2]), 1./(ko[-1]))
    print "open wait time from I: {:.1f} ms".format(1./(kim))
    print "close wait time from O: {:.3f} ms".format(1./(kom + ki[-1]))
    print "Fraction of wait time going to I from O: {:.3f} ms".format(ki[-1]/(kom + ki[-1]))
