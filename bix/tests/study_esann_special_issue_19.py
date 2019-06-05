#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 3 08:58:46 2019

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

import numpy as np
from bix.evaluation.gridsearch import GridSearch
from bix.evaluation.crossvalidation import CrossValidation
from bix.classifiers.arslvq import ARSLVQ
from skmultiflow.trees import HAT
from skmultiflow.lazy import SAMKNN, KNN
from skmultiflow.meta import OzaBagging, OzaBaggingAdwin, AdaptiveRandomForest

def test_parameter_grid_search_arslvq():
    grid = {"sigma": np.append(1, np.arange(
        2, 11, 2)), "prototypes_per_class": np.append(1, np.arange(2, 11, 2)),
        "gamma": np.array([0.7, 0.9, 0.999]),
        "confidence": np.array([0.01, 0.001]),
        "window_size": np.array([100, 200, 300, 800])}
    clf = ARSLVQ()
    gs = GridSearch(clf=clf, grid=grid, max_samples=50000)
    gs.streams = gs.init_real_world() + gs.init_standard_streams()  + gs.init_reoccuring_standard_streams()
    gs.search()
    gs.save_summary()
    
def test_grid():
    clfs = [
            OzaBagging(base_estimator=KNN()), 
            OzaBaggingAdwin(base_estimator=KNN()), 
            AdaptiveRandomForest(), 
            SAMKNN(),
            HAT()
    ]
    cv = CrossValidation(clfs=clfs, max_samples=1000000, test_size=1)
    cv.streams = cv.init_real_world() + cv.init_standard_streams()  + cv.init_reoccuring_streams()
    cv.test()
    cv.save_summary()

if __name__ == "__main__":
    test_parameter_grid_search_arslvq()