import unittest
import numpy as np
import glob, os
import scipy.io as sio
from bix.transfer_learning.nys_basis_transfer import Nystöm_Basis_Transfer
from sklearn import preprocessing
from sklearn import neighbors
class TestNBT(unittest.TestCase):


    def test_source_data(self):
        nbt = Nystöm_Basis_Transfer()
        with self.assertRaises(ValueError):
            nbt.data_augmentation(5,5,np.array([]))

    def test_size_data(self):
        nbt = Nystöm_Basis_Transfer()
        with self.assertRaises(ValueError):
            nbt.data_augmentation(np.array([]),[],np.array([]))

    def test_label_data(self):
        nbt = Nystöm_Basis_Transfer()
        with self.assertRaises(ValueError):
            nbt.data_augmentation(np.array([]),5,5)

    def test_source_greater_target(self):
        import glob, os
        import scipy.io as sio

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(os.path.join("..","..","datasets","domain_adaptation","reuters"))
        errors = sio.loadmat("org_vs_people_1.mat")
        X = np.asarray(errors["Xt"].T.todense())
        Z = np.asarray(errors["Xs"].T.todense())
        Ys = np.asarray(errors["Ys"].todense())
        Yt = np.asarray(errors["Yt"].todense())
        nbt = Nystöm_Basis_Transfer()
        Ys,Z = nbt.data_augmentation(Z,X.shape[0],Ys)

        self.assertTrue(X.shape[0]==Z.shape[0]==Ys.shape[0],msg='Sample Sizes must be equal')

        
    def test_source_smaller_target(self):
        import glob, os
        import scipy.io as sio

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(os.path.join("..","..","datasets","domain_adaptation","reuters"))
        errors = sio.loadmat("org_vs_people_2.mat")
        X = np.asarray(errors["Xt"].T.todense())
        Z = np.asarray(errors["Xs"].T.todense())
        Ys = np.asarray(errors["Ys"].todense())
        Yt = np.asarray(errors["Yt"].todense())
        nbt = Nystöm_Basis_Transfer()
        Ys,Z = nbt.data_augmentation(Z,X.shape[0],Ys)

        self.assertTrue(X.shape[0]==Z.shape[0]==Ys.shape[0],msg='Sample Sizes must be equal')

    def test_landmark(self):
        nbt = Nystöm_Basis_Transfer()
        with self.assertRaises(ValueError):
            nbt.nys_basis_transfer(np.array([]),np.array([]),5.0)
        with self.assertRaises(ValueError):
            nbt.nys_basis_transfer(np.array([]),np.array([]),-3)
        with self.assertRaises(ValueError):
            nbt.nys_basis_transfer(np.array([]),np.array([]),-2.0)

    def test_source(self):
        nbt = Nystöm_Basis_Transfer()
        with self.assertRaises(ValueError):
            nbt.nys_basis_transfer(1,np.array([]),5)

    def test_target(self):
        nbt = Nystöm_Basis_Transfer()
        with self.assertRaises(ValueError):
            nbt.nys_basis_transfer(np.array([]),1,5)

    def test_dimension_reduction(self):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(os.path.join("..","..","datasets","domain_adaptation","reuters"))
        errors = sio.loadmat("org_vs_people_1.mat")
        X = np.asarray(errors["Xt"].T.todense())
        Z = np.asarray(errors["Xs"].T.todense())
        Ys = np.asarray(errors["Ys"].todense())
        Yt = np.asarray(errors["Yt"].todense())
        nbt = Nystöm_Basis_Transfer()
        X,Z = nbt.nys_basis_transfer(X,Z,600)
        self.assertTrue(X.shape[1]==Z.shape[1],msg='Size of dimensions must be equal')

    def test_transfer_learning(self):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(os.path.join("..","..","datasets","domain_adaptation","reuters"))
        errors = sio.loadmat("org_vs_people_1.mat")
        X = np.asarray(errors["Xt"].T.todense())
        Z = np.asarray(errors["Xs"].T.todense())
        Ys = np.asarray(errors["Ys"].todense())
        Yt = np.asarray(errors["Yt"].todense())

        X = preprocessing.scale(X)
        Z = preprocessing.scale(Z)
        clf = neighbors.KNeighborsClassifier(10)
        clf.fit(Z, Ys)
        predicton =  clf.predict(X)
        acc_without = np.sum(predicton==np.squeeze(Yt)) / Yt.shape[0]
        nbt = Nystöm_Basis_Transfer()
        Ys,Z = nbt.data_augmentation(Z,X.shape[0],Ys)
        X,Z = nbt.nys_basis_transfer(X,Z,600)
        clf = neighbors.KNeighborsClassifier(2)
        clf.fit(Z, Ys)
        predicton =  clf.predict(X)
        acc_transfer = np.sum(predicton==np.squeeze(Yt)) / Yt.shape[0]

        self.assertTrue(acc_without < acc_transfer,msg='Transfer accuracy should be higher!')

    def test_multi_label_source_greater(self):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(os.path.join("..","..","datasets","domain_adaptation","OfficeCaltech"))
        amazon = sio.loadmat("amazon_SURF_L10.mat")
        Z = np.asarray(amazon["fts"])
        Ys = np.asarray(amazon["labels"])

        dslr = sio.loadmat("dslr_SURF_L10.mat")

        X = np.asarray(dslr["fts"])
        Yt = np.asarray(dslr["labels"])
        nbt = Nystöm_Basis_Transfer()
        Ys,Z = nbt.data_augmentation(Z,X.shape[0],Ys)
        unique_labels = len(np.unique(Ys))
        self.assertTrue(X.shape[0]==Z.shape[0]==Ys.shape[0] and unique_labels==10)


    def test_multi_label_source_smaller(self):
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
        unique_labels = len(np.unique(Ys))
        self.assertTrue(X.shape[0]==Z.shape[0]==Ys.shape[0] and unique_labels==10)

if __name__ == "__main__":
    unittest.main()
    
  
    