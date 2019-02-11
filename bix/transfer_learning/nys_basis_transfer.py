import numpy as np


class Nystöm_Basis_Transfer():
    """
    Nyström Basis Transfer Service Class
    
    Functions
    ----------
    nys_basis_transfer: Transfer Basis from Target to Source Domain.
    data_augmentation: Augmentation of data by removing or upsampling of source data

    Examples
    --------
    >>> #Imports
    >>> import glob, os
    >>> import scipy.io as sio
    >>> from sklearn import preprocessing
    >>> from sklearn import neighbors
    >>> os.chdir(os.path.dirname(os.path.abspath(__file__)))
    >>> os.chdir(os.path.join("..","..","datasets","domain_adaptation","reuters"))
    >>> errors = sio.loadmat("org_vs_people_1.mat")
    >>> X = np.asarray(errors["Xt"].T.todense())
    >>> Z = np.asarray(errors["Xs"].T.todense())
    >>> Ys = np.asarray(errors["Ys"].todense())
    >>> Yt = np.asarray(errors["Yt"].todense())

    >>> X = preprocessing.scale(X)
    >>> Z = preprocessing.scale(Z)
    >>> clf = neighbors.KNeighborsClassifier(10)
    >>> clf.fit(Z, Ys)
    >>> predicton =  clf.predict(X)
    >>> acc = np.sum(predicton==np.squeeze(Yt)) / Yt.shape[0]
    >>> print(acc)
    >>> nbt = Nystöm_Basis_Transfer()
    >>> Ys,Z = nbt.data_augmentation(Z,X.shape[0],Ys)
    >>> X,Z = nbt.nys_basis_transfer(X,Z,600)
    >>> clf = neighbors.KNeighborsClassifier(2)
    >>> clf.fit(Z, Ys)
    >>> predicton =  clf.predict(X)
    >>> acc = np.sum(predicton==np.squeeze(Yt)) / Yt.shape[0]
    >>> print(acc)
    """

    def __init__(self):
        # TO be filled
        pass

    def nys_basis_transfer(self,X,Z,landmarks=10):
        """
        Nyström Basis Transfer
        Transfers Basis of X to Z obtained by Nyström SVD
        Implicit dimensionality reduction
        Applications in domain adaptation or transfer learning
        Parameters.
        Note target,source are order sensitiv.
        ----------
        X : Target Matrix, where classifier is trained on
        Z : Source Matrix, where classifier is trained on
        landmarks : Positive integer as number of landmarks
        
        Returns
        ----------
        X : Reduced Target Matrix
        Z : Reduced approximated Source Matrix

        """
        if type(X) is not np.ndarray or type(Z) is not np.ndarray:
            raise ValueError("Numpy Arrays must be given!")
        if type(landmarks) is not int or landmarks < 1:
            raise ValueError("Positive integer number must given!")
        landmarks = np.min(list(X.shape)+list(Z.shape)+[landmarks])
        max_idx = np.min(list(X.shape)+list(Z.shape))
        idx = np.random.randint(0,max_idx-1,landmarks)
        A = X[np.ix_(idx,idx)]
        B = X[0:landmarks,landmarks:]
        F = X[landmarks:,0:landmarks]
        C = X[landmarks:,landmarks:]
        U, S, H = np.linalg.svd(A, full_matrices=True)
        S = np.diag(S)

        U_k = np.concatenate([U, np.matmul(np.matmul(F,H),np.linalg.pinv(S))])
        V_k = np.concatenate([H, np.matmul(np.matmul(B.T,U),np.linalg.pinv(S))])
        X = np.matmul(U_k,S)

        A = Z[np.ix_(idx,idx)]
        D = np.linalg.svd(A, full_matrices=True,compute_uv=False)
        Z = np.matmul(U_k,np.diag(D))
        return X,Z

    def data_augmentation(self,Z,required_size,Y):
        """
        Data Augmentation
        Upsampling if Z smaller as required_size via multivariate gaussian mixture
        Downsampling if Z greater as required_size via uniform removal

        Note both are class-wise with goal to harmonize class counts
        ----------
        Z : Matrix, where classifier is trained on
        required_size : Size to which Z is reduced or extended 
        Y : Label vector, which is reduced or extended like Z
        
        Returns
        ----------
        X : Augmented Z
        Z : Augmented Y

        """
        if type(Z) is not np.ndarray or type(required_size) is not int or type(Y) is not np.ndarray:
            raise ValueError("Numpy Arrays must be given!")
        if Z.shape[0] == required_size:
            return Y,Z
        
        _, idx = np.unique(Y, return_index=True)
        C = Y[np.sort(idx)].flatten().tolist()
        size_c = len(C)
        if Z.shape[0] < required_size:
            print("Source smaller target")
            data = np.empty((0,Z.shape[1]))
            label = np.empty((0,1))
            diff = required_size - Z.shape[0]
            sample_size = int(np.floor(diff/size_c))
            for c in C:
                print(c)
                indexes = np.where(Y[Y==c])
                class_data = Z[indexes,:][0]
                m = np.mean(class_data,0) 
                sd = np.var(class_data,0)
                sample_size = sample_size if c !=C[-1] else sample_size+np.mod(diff,size_c)
                augmentation_data =np.vstack([np.random.normal(m, sd, len(m)) for i in range(sample_size)])
                data =np.concatenate([data,class_data,augmentation_data])
                label = np.concatenate([label,np.ones((class_data.shape[0]+sample_size,1))*c])
            
        if Z.shape[0] > required_size:
            print("Source greater target")
            data = np.empty((0,Z.shape[1]))
            label = np.empty((0,1))
            sample_size = int(np.floor(required_size/size_c))
            for c in C:
                indexes = np.where(Y[Y==c])[0]
                class_data = Z[indexes,:]
                if len(indexes) > sample_size:
                    sample_size = sample_size if c !=C[-1] else np.abs(data.shape[0]-required_size)
                    y = np.random.choice(class_data.shape[0],sample_size)
                    class_data = class_data[y,:]
                data =np.concatenate([data,class_data])
                label = np.concatenate([label,np.ones((class_data.shape[0],1))*c])
        Z = data
        Y = label
        return Y,Z

if __name__ == "__main__":

    import glob, os
    import scipy.io as sio
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(os.path.join("..","..","datasets","domain_adaptation","OfficeCaltech"))
    amazon = sio.loadmat("amazon_SURF_L10.mat")
    X = np.asarray(amazon["fts"])
    Xt = np.asarray(amazon["labels"])

    dslr = sio.loadmat("dslr_SURF_L10.mat")

    Z = np.asarray(dslr["fts"])
    Ys = np.asarray(dslr["labels"])
    nbt = Nystöm_Basis_Transfer()
    Ys,Z = nbt.data_augmentation(Z,X.shape[0],Ys)
