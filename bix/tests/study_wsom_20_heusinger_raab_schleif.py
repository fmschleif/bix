import numpy as np
from bix.evaluation.crossvalidation import CrossValidation
from bix.evaluation.gridsearch import GridSearch
from bix.classifiers.rslvq import RSLVQ
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest
from skmultiflow.trees.hoeffding_adaptive_tree import HAT
from skmultiflow.lazy.knn import KNN
from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin
from skmultiflow.lazy.sam_knn import SAMKNN
from bix.classifiers.adaptive_rslvq import ARSLVQ
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.meta.oza_bagging import OzaBagging
from skmultiflow.data.led_generator_drift import LEDGeneratorDrift
from skmultiflow.data.concept_drift_stream import ConceptDriftStream

def test_parameter_grid_search_rslvq():
    grid = {"sigma": np.arange(
        2, 11, 2), "prototypes_per_class": np.arange(2, 11, 2)}
    clf = RSLVQ()
    gs = GridSearch(clf=clf, grid=grid, max_samples=50000)
    gs.search()
    gs.save_summary()


def test_parameter_grid_search_arslvq():
    grid = {"sigma": np.arange(
        2, 11, 2), "prototypes_per_class": np.arange(2, 11, 2)}
    clf = ARSLVQ(gradient_descent="Adadelta")
    gs = GridSearch(clf=clf, grid=grid, max_samples=50000)
    gs.search()
    gs.save_summary()


# def test_grid():
#     clfs = [ARSLVQ(gradient_descent="Adadelta"), RSLVQ(), HoeffdingTree(), HAT(), OzaBagging(
#         base_estimator=KNN()), OzaBaggingAdwin(base_estimator=KNN()), AdaptiveRandomForest(), SAMKNN()]
#     cv = CrossValidation(clfs=clfs, max_samples=1000000, test_size=5)
#     cv.streams = cv.init_standard_streams() + cv.init_real_world() + \
#         cv.init_reoccuring_streams()
#     cv.test()
#     cv.save_summary()
# def test_grid():
#     clfs = [OzaBagging(
#         base_estimator=KNN()), OzaBaggingAdwin(base_estimator=KNN()), AdaptiveRandomForest(), SAMKNN()]
#     cv = CrossValidation(clfs=clfs, max_samples=1000000, test_size=3)
#     cv.streams = cv.init_standard_streams() + cv.init_real_world()
#     cv.test()
#     cv.save_summary()


def test_grid():
    clfs = [OzaBagging(
    base_estimator=KNN()), OzaBaggingAdwin(base_estimator=KNN()), AdaptiveRandomForest(), SAMKNN()]
    cv = CrossValidation(clfs=clfs, max_samples=1000000, test_size=1)
    cv.streams = [ConceptDriftStream(stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                            drift_stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=7),
                            random_state=None,
                            alpha=90.0, # angle of change grade 0 - 90
                            position=250000,
                            width=1),ConceptDriftStream(stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                            drift_stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=7),
                            random_state=None,
                            alpha=90.0, # angle of change grade 0 - 90
                            position=250000,
                            width=50000)]
    cv.test()
    cv.save_summary()
if __name__ == "__main__":

    # test_parameter_grid_search_arslvq()
    # test_parameter_grid_search_rslvq()
    test_grid()
