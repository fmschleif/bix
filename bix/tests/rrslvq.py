
from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.lazy.knn import KNN
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from bix.classifiers.rrslvq import RRSLVQ
import unittest
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score

class TestRRSLVQ(unittest.TestCase):
    
    def setUp(self):
        # is executed first
        X, y = load_wine(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.33, random_state=42)

    def test_sigma(self):
        with self.assertRaises(ValueError):
            RRSLVQ(sigma=-1)
        RRSLVQ(sigma=1)

    def test_prototypes(self):
        with self.assertRaises(ValueError):
            RRSLVQ(prototypes_per_class=0)
        RRSLVQ(prototypes_per_class=10)

    def test_confidence(self):
        with self.assertRaises(ValueError):
            RRSLVQ(confidence=-1)
        RRSLVQ(confidence=0.1)
        with self.assertRaises(ValueError):
            RRSLVQ(confidence=1.1)
        RRSLVQ(confidence=0.1)

    def test_drift_detector(self):
        with self.assertRaises(ValueError):
            RRSLVQ(drift_detector="XX")
        RRSLVQ(drift_detector="KS")

    def test_accuracy_data(self):

        clf = RRSLVQ(sigma=10, prototypes_per_class=5)
        clf.partial_fit(self.X_train, self.y_train)

        y_pred = clf.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        kappa = cohen_kappa_score(self.y_test, y_pred)

        self.assertTrue(acc >= 0.5, msg='Accuracy has to be greater than 0.5, \
                        but was {}'.format(acc))
        self.assertTrue(kappa >= 0.3, msg='Kappa has to be greater than 0.3, \
                        but was {}'.format(kappa))

    def test_stream(self):
        stream = SEAGenerator(classification_function = 2, random_state = 112, balance_classes = False, noise_percentage = 0.28)     
        stream.prepare_for_use()

        evaluator = EvaluatePrequential(show_plot=False,max_samples=5000, 
                restart_stream=True,batch_size=10,metrics=['kappa', 'kappa_m', 'accuracy']) 

        evaluator.evaluate(stream=stream, model=RRSLVQ(prototypes_per_class=4,sigma=10))

        measurements = np.asarray(evaluator.get_measurements()[0])[0]
        self.assertIsNotNone(eval)
        self.assertTrue(measurements.get_accuracy() >= 0.5,
                        msg='Accuracy was {} but has to be greater than 0.5'.
                        format(measurements.get_accuracy()))


if __name__ == "__main__":
    unittest.main()
    
  
    