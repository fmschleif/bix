import unittest
import numpy as np
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


if __name__ == '__main__':
    unittest.main()
