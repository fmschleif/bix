import numpy as np
from bix.classifiers.rrslvq import RRSLVQ
from bix.evaluation.crossvalidation import CrossValidation
from bix.evaluation.gridsearch import GridSearch
from bix.classifiers.rslvq import RSLVQ 
from skmultiflow.bayes import NaiveBayes
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest
from skmultiflow.trees.hoeffding_adaptive_tree import HAT
from skmultiflow.lazy.knn import KNN
from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin
from skmultiflow.lazy.sam_knn import SAMKNN
from bix.classifiers.adaptive_rslvq import ARSLVQ
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.meta.oza_bagging import OzaBagging
def test_parameter_grid_search_arslvq(self):
    grid = {"sigma": np.arange(2,3,2), "prototypes_per_class": np.arange(2,3,2)}
    clf = ARSLVQ()
    gs = GridSearch(clf=clf,grid=grid,max_samples=1000)
    gs.search()
    gs.save_summary()

def test_parameter_grid_search_rslvq(self):
    grid = {"sigma": np.arange(2,3,2), "prototypes_per_class": np.arange(2,3,2)}
    clf = RSLVQ()
    gs = GridSearch(clf=clf,grid=grid,max_samples=1000)
    gs.search()
    gs.save_summary()

def test_grid():
    clfs = [ARSLVQ(gradient_descent="Adadelta"),RSLVQ(),HoeffdingTree(),HAT(),OzaBagging(base_estimator=KNN()),OzaBaggingAdwin(base_estimator=KNN()),AdaptiveRandomForest(),SAMKNN()]
    cv = CrossValidation(clfs=clfs,max_samples=500,test_size=1)
    cv.streams = cv.init_standard_streams()+cv.init_real_world()+ cv.init_reoccuring_streams()
    cv.test()
    cv.save_summary()
    print("here")


if __name__ == "__main__":  
  test_grid()
