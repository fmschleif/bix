import numpy as np
from bix.evaluation.crossvalidation import CrossValidation
from bix.classifiers.rrslvq import RRSLVQ
from skmultiflow.bayes.naive_bayes import NaiveBayes
import unittest

class TESTCROSSVALIDATION(unittest.TestCase):
    
    def test_save_summary(self):
        with self.assertRaises(ValueError):
            cv = CrossValidation(clfs=[RRSLVQ(),NaiveBayes()],max_samples=500,test_size=1)
            cv.save_summary()

    def test_grid_search(self):  
        cv = CrossValidation(clfs=[RRSLVQ(),NaiveBayes()],max_samples=500,test_size=1)
        cv.streams = cv.init_reoccuring_streams()
        cv.test()
        cv.save_summary()
        self.assertIsNotNone(cv.result)

if __name__ == '__main__':
    unittest.main()