from skmultiflow.data.mixed_generator import MIXEDGenerator
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from bix.data.reoccuringdriftstream import ReoccuringDriftStream
from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin
from skmultiflow.lazy.knn import KNN
import unittest
import numpy as np

class TESTGRIDSEARCH(unittest.TestCase):
    """TESTCROSSVALIDATION"""
    
    def test_width(self):
        with self.assertRaises(ValueError):
            ReoccuringDriftStream(width=-1)
        ReoccuringDriftStream(width=1)

    def test_alpha(self):
        with self.assertRaises(ValueError):
            ReoccuringDriftStream(alpha=-1)
        with self.assertRaises(ValueError):
            ReoccuringDriftStream(alpha=91)
        ReoccuringDriftStream(alpha=1)

    def test_pause(self):
        with self.assertRaises(ValueError):
            ReoccuringDriftStream(pause=-1)
        ReoccuringDriftStream(pause=1)


    def test_reoccuring(self):
        s1 = MIXEDGenerator(classification_function = 1, random_state= 112, balance_classes = False)
        s2 = MIXEDGenerator(classification_function = 0, random_state= 112, balance_classes = False)
        stream = ReoccuringDriftStream(stream=s1,
                                drift_stream=s2,
                                random_state=None,
                                alpha=90.0, # angle of change grade 0 - 90
                                position=2000,
                                width=500)
        stream.prepare_for_use()
        evaluator = EvaluatePrequential(show_plot=False,batch_size=10,
                                        max_samples=1000,
                                        metrics=['accuracy', 'kappa_t', 'kappa_m', 'kappa'],    
                                        output_file=None)
        eval = evaluator.evaluate(stream=stream, model=OzaBaggingAdwin(base_estimator=KNN()))
        

        measurements = np.asarray(evaluator.get_measurements()[0])[0]
        
        self.assertIsNotNone(eval)
        self.assertTrue(measurements.get_accuracy() >= 0.6,
                        msg='Accuracy was {} but has to be greater than 0.6'.
                        format(measurements.get_accuracy()))

if __name__ == "__main__":
    unittest.main()
    
  