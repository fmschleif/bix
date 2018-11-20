from __future__ import division

import numpy as np
from bix.classifiers.rrslvq import RRSLVQ
from bix.utils.crossvalidation import CrossValidation
from bix.utils.gridsearch import GridSearch
from bix.classifiers.rslvq import RSLVQ 
from skmultiflow.bayes import NaiveBayes
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest
from skmultiflow.trees.hoeffding_adaptive_tree import HAT



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