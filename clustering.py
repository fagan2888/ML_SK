import math# import math.log, math.pi, math.lgamma
import numpy as np

# Calculate the posterior math.loglikelihood of a hard assignment of gaussian linear regression mixtures.
def loglikelihood(X, Y, XYlabels, assignments, labels, K):
    N = X.size

    ll = math.lgamma(0.5*K) - 0.5*K*math.log(math.pi) - math.lgamma(N+0.5*K)

    for k in xrange(K):
        idx = np.in1d(XYlabels,labels[assignments == k])
        nk = np.sum(idx)
        x = X[idx]
        y = Y[idx]

        x2 = np.sum(x*x)
        y2 = np.sum(y*y)
        xy = np.sum(x*y)
        x1 = np.sum(x)
        y1 = np.sum(y)

        alpha = nk*x2*y2 - nk*xy**2 - x1**2*y2 - x2*y1**2 + 2*x1*xy*y1
        beta = (0.5*nk-1.5)*math.log(nk*x2-x1**2) + math.lgamma(0.5*nk - 1)

        ll += math.lgamma(nk+0.5) + math.log(beta)
    return ll
