#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 3 08:58:46 2019

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""
import os
import numpy as np
from bix.evaluation.gridsearch import GridSearch
from bix.evaluation.crossvalidation import CrossValidation
from bix.classifiers.arslvq import ARSLVQ
from skmultiflow.trees import HAT
from skmultiflow.lazy import SAMKNN, KNN
from skmultiflow.meta import OzaBagging, OzaBaggingAdwin, AdaptiveRandomForest
from skmultiflow.data import LEDGeneratorDrift
from bix.data.reoccuringdriftstream import ReoccuringDriftStream
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.data.file_stream import FileStream
def test_gridsearch_via_cv():
    grid = {"sigma": np.append(1, np.arange(
        2, 11, 2)), "prototypes_per_class": np.append(1, np.arange(2, 11, 2)),
        "gamma": np.array([0.7, 0.9, 0.999]),
        "confidence": np.array([0.01, 0.001]),
        "window_size": np.array([100, 200, 300, 800])}
    grid = {"sigma": np.append(1, np.arange(
        2, 3, 2)), "prototypes_per_class": np.append(1, np.arange(2, 3, 2))}

    clf = ARSLVQ()
    cv = GridSearch([clf],max_samples=500)
    cv.streams = cv.init_standard_streams()
    cv.parameter_grid_search(grid)

if __name__ == "__main__":
   # test_parameter_grid_search_arslvq()
   #test_missing_streams()
    test_gridsearch_via_cv()