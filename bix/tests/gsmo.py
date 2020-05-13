import unittest
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
from sklearn.svm import SVC
from bix.utils.gsmo_solver import GSMO


class TESTGSMO(unittest.TestCase):
    def test_init_x_1d(self):
        # Arrange
        C = np.array([-1, 1, -1])
        d = 3

        # Act
        gsmo_solver = GSMO(A=np.zeros((3, 3)), b=np.zeros((3, 1)), C=C, d=d, r=0, R=5)

        # Assert
        # Cx = d
        result = C.dot(gsmo_solver.x)
        self.assertAlmostEqual(d, result)

    def test_init_x_2d(self):
        # Arrange
        C = np.array([[-1, 1, -1], [2, 0, 3]])
        d = np.array([3, 1])

        # Act
        gsmo_solver = GSMO(A=np.zeros((3, 3)), b=np.zeros((3, 1)), C=C, d=d, r=0, R=5)

        # Assert
        # Cx = d
        result = C.dot(gsmo_solver.x)
        np.testing.assert_almost_equal(d, result)

    def test_init_x_valueError(self):
        # Arrange
        C = np.array([[-1, 1, 1], [-1, 1, 1]])
        d = np.array([2, 3])

        # Act
        with self.assertRaises(ValueError):
            GSMO(A=np.zeros((3, 3)), b=np.zeros((3, 1)), C=C, d=d, r=0, R=5)

    def test_init_small_svm(self):
        # Arrange
        pwd = os.path.dirname(os.path.abspath(__file__))
        test_data_file = os.path.join(pwd, "small_svm_problem_data.csv")
        data = pd.read_csv(test_data_file, delimiter=',')
        print(data)
        A = np.zeros((data.shape[0], data.shape[0]))
        points = data[['X', 'Y']]
        y = data['Label']
        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                A[i, j] = y.iloc[i] * y.iloc[j] * points.iloc[i].dot(points.iloc[j])

        A = (-0.5) * A
        b = np.ones((1, A.shape[0]))
        C = y.to_numpy()
        d = 0

        # Act
        gsmo_solver = GSMO(A, b, C, d, 0, 100)

        # Assert
        # Cx = d
        result = C.dot(gsmo_solver.x)
        np.testing.assert_almost_equal(d, result)

    def test_solve_small_svm(self):
        # Arrange
        pwd = os.path.dirname(os.path.abspath(__file__))
        test_data_file = os.path.join(pwd, "small_svm_problem_data.csv")
        data = pd.read_csv(test_data_file, delimiter=',')
        print(data)
        A = np.zeros((data.shape[0], data.shape[0]))
        points = data[['X', 'Y']]
        y = data['Label']
        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                A[i, j] = y.iloc[i] * y.iloc[j] * points.iloc[i].dot(points.iloc[j])

        A = (0.5) * A
        b = -np.ones((A.shape[0],))
        C = y.to_numpy()
        C_t = C.reshape((1, C.shape[0]))
        d = 0
        gsmo_solver = GSMO(A, b, C_t, d, 0, 1, step_size=0.1)

        # Act
        print("SMO")
        gsmo_solver.solve()
        print(gsmo_solver.x.round(3))
        # Assert
        # Cx = d
        result = C.dot(gsmo_solver.x)
        np.testing.assert_almost_equal(d, result)

        fun = lambda x, H, f: x.transpose().dot(H).dot(x) + f.transpose().dot(x)
        bnds = tuple([(0, 1) for i in range(A.shape[0])])
        constr = ({'type': 'eq', 'args': C_t, 'fun': lambda x, c: c.transpose().dot(x)})
        res = minimize(fun, np.ones((A.shape[0],)), args=(A, b), bounds=bnds, constraints=constr)
        print("MINIMIZE")
        print(res.x)

        print("SVC")
        clf = SVC(C=1, kernel='linear')
        clf.fit(points, y)
        print(clf.dual_coef_)
        print(clf.support_)


if __name__ == '__main__':
    unittest.main()
