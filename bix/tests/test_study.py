from __future__ import division

import copy
import datetime
import itertools
import math
import os
import sys
from random import random as rnd

import numpy as np
from bix.classifier.rrslvq import RRSLVQ
from CrossValidation import CrossValidation
from GridSearch import GridSearch
from rrslvq import RRSLVQ as rRSLVQ
from rslvq import RSLVQ as bRSLVQ
from rslvq_stream import RSLVQ as adaRSLVQ
from skmultiflow.bayes import NaiveBayes
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest
from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin
from skmultiflow.trees.hoeffding_adaptive_tree import HAT

from stream_rslvq import RSLVQ as sRSLVQ


def parameter_grid_search_test():
    grid = {"sigma": np.arange(2,3,2), "prototypes_per_class": np.arange(2,3,2)}
    clf = RRSLVQ()
    gs = GridSearch(clf=clf,grid=grid,max_samples=1000)
    gs.search()
    gs.save_summary()

def cross_validation_test():
    clfs = [RRSLVQ(),RSLVQ(),HAT(),AdaptiveRandomForest(),NaiveBayes()]
    
    cv = CrossValidation(clfs=clfs,max_samples=1000,test_size=1)
    cv.test()
    cv.save_summary()

if __name__ == "__main__":  
    parameter_grid_search_test()
    cross_validation_test()
