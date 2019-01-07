from __future__ import division

import numpy as np
from bix.classifiers.rrslvq import RRSLVQ
from bix.evaluation.crossvalidation import CrossValidation
from bix.evaluation.gridsearch import GridSearch
from bix.classifiers.rslvq import RSLVQ 
from skmultiflow.bayes import NaiveBayes
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest
from skmultiflow.trees.hoeffding_adaptive_tree import HAT
import os
import unittest
class TestStudy(unittest.TestCase):

    def test_parameter_grid_search_test(self):
        grid = {"sigma": np.arange(2,3,2), "prototypes_per_class": np.arange(2,3,2)}
        clf = RRSLVQ()
        gs = GridSearch(clf=clf,grid=grid,max_samples=1000)
        gs.search()
        gs.save_summary()

    def test_cross_validation_real_world_test(self):
        clfs = [RRSLVQ(prototypes_per_class=2), HAT()]
        
        cv = CrossValidation(clfs=clfs,max_samples=500,test_size=1)
        cv.streams = cv.init_real_world()
        cv.test()
        cv.save_summary()

    def test_cross_validation_standard_test(self):
        clfs = [RRSLVQ(prototypes_per_class=2), HAT()]
        
        cv = CrossValidation(clfs=clfs,max_samples=500,test_size=1)
        cv.test()
        cv.save_summary()

    def test_cross_validation_reoccuring_test(self):
        clfs = [RRSLVQ(prototypes_per_class=2), HAT()]
        
        cv = CrossValidation(clfs=clfs,max_samples=500,test_size=1)
        cv.streams= cv.init_reoccuring_streams()
        cv.test()
        cv.save_summary()

if __name__ == "__main__":  
    unittest.main()