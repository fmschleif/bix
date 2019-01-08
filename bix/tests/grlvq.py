#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 09:02:55 2018

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

import unittest
from bix.classifiers.grlvq import GRLVQ
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
import numpy as np

class TestGRLVQ(unittest.TestCase):
    # TODO: implement + adjust tests as soon as method works fine
        
    def test_correct_init(self):
        GRLVQ(prototypes_per_class=2, regularization=0.0,
              beta=2, C=None)
    
    def test_accuracy_stream(self):
        stream = SEAGenerator(random_state=42)
        stream.prepare_for_use()

        clf = GRLVQ(prototypes_per_class=2, regularization=0.0,
              beta=2, C=None)

        evaluator = EvaluatePrequential(pretrain_size=1, show_plot=False, max_samples=20000,
                                        batch_size=1)

        evaluator.evaluate(stream, clf, model_names=['GRLVQ'])

        measurements = np.asarray(evaluator.get_measurements()[0])[0]

        self.assertTrue(measurements.get_accuracy() >= 0.7,
                        msg='Accuracy was {} but has to be greater than 0.7'.
                        format(measurements.get_accuracy()))
        self.assertTrue(measurements.get_kappa() >= 0.3,
                        msg='Kappa was {} but has to be greater than 0.3'.
                        format(measurements.get_kappa()))
        
if __name__ == '__main__':
    unittest.main()