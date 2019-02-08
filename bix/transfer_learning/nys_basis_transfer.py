import numpy as np

def nys_basis_transfer(X,Z,landmarks=10):
    """
    Nyström Basis Transfer
    Transfers Basis of X to Z obtained by Nyström SVD
    Implicit dimensionality reduction
    Applications in domain adaptation or transfer learning
    Parameters
    ----------
    X : Target Matrix, where classifier is trained on
    Z : Source Matrix, where classifier is trained on
    landmarks : Positive integer as number of landmarks
    
    Returns
    ----------
    X : Reduced Target Matrix
    Z : Reduced approximated Source Matrix

    Examples
    --------
    >>> #Imports
    """
    if type(X) is not np.ndarray or type(Z) is not np.ndarray:
        raise ValueError("Numpy Arrays must be given!")
    if type(landmarks) is not int or landmarks < 1:
         raise ValueError("Positive integer number must given!")
    landmarks = np.min(list(X.shape)+list(Z.shape)+[landmarks])
    max_idx = np.min(list(X.shape)+list(Z.shape))
    idx = np.random.randint(0,max_idx-1,landmarks)
    A = X[np.ix_(idx,idx)]
    B = X[0:landmarks,landmarks+1:]
    F = X[landmarks+1:,0:landmarks]
    C = X[landmarks+1:,landmarks+1:]
    U, S, H = np.linalg.svd(A, full_matrices=True)
    S = np.diag(S)

    U_k = np.concatenate([U, np.matmul(np.matmul(F,H),np.linalg.pinv(S))])
    V_k = np.concatenate([H, np.matmul(np.matmul(B.T,U),np.linalg.pinv(S))])
    X = np.matmul(U_k,S)

    A = Z[np.ix_(idx,idx)]
    D = np.linalg.svd(A, full_matrices=True,compute_uv=False)
    Z = np.matmul(U_k,np.diag(D))
    return X,Z

if __name__ == "__main__":
    import glob, os
    import scipy.io as sio
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(os.path.join("..","..","datasets","domain_adaptation","reuters"))
    errors = sio.loadmat("org_vs_people_1.mat")
    X = np.asarray(errors["Xt"].T.todense())
    Z = np.asarray(errors["Xs"].T.todense())

    nys_basis_transfer(X,Z,1000)