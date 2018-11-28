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

def test_grid():
    clfs = [RRSLVQ(),RSLVQ(),HAT(),OzaBaggingAdwin(base_estimator=KNN()),AdaptiveRandomForest(),SAMKNN()]
    cv = CrossValidation(clfs=clfs,max_samples=1000000,test_size=1)
    cv.init_reoccuring_streams()
    cv.test()
    cv.save_summary()
    print("here")

if __name__ == "__main__":  
  test_grid()
