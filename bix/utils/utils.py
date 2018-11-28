import numpy as np

"""
Utils script for various repeating actions
"""
def incrementalMean(x,xm=None,n=None,first = True):
    """ calculates incremental mean 
    """
    if n is None:
        n = 1
    if xm is None:
        xm = 0
    xm = xm + (1/n)*(x-xm)
    n+=1
    return (xm,n)

def expandingMean(x,alpha = 0.9, xm=None,n=None):
    """ calculates expanding mean
    """
    if n is None:
        n = 1
    if xm is None:
        xm = np.zeros(x.size)
    xm = alpha*xm + (1-alpha)* x
    n+=1
    return (xm,n) 
