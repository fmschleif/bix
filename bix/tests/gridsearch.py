import numpy as np
from bix.evaluation.gridsearch import GridSearch
from bix.classifiers.rrslvq import RRSLVQ
import unittest

class TESTGRIDSEARCH(unittest.TestCase):

    def test_save_summary(self):
        with self.assertRaises(ValueError):
            grid = {"sigma": np.arange(2, 3, 2), "prototypes_per_class": np.arange(2, 3, 2)}
            gs = GridSearch(clf=RRSLVQ(),grid=grid,max_samples=300)
            gs.save_summary()

    def test_grid_search(self):
        grid = {"sigma": np.arange(2, 3, 2), "prototypes_per_class": np.arange(2, 3, 2)}
        gs = GridSearch(clf=RRSLVQ(), grid=grid, max_samples=300)
        gs.search()
        gs.save_summary()
        self.assertIsNotNone(gs.best_runs)

if __name__ == '__main__':
    unittest.main()