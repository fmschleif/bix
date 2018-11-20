#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:27:33 2018

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

import unittest
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.data.sea_generator import SEAGenerator
from bix.classifiers.rslvq import RSLVQ
import numpy as np
from bix.classifiers.adaptive_rslvq import RSLVQ as ARSLVQ


class TestRSLVQ(unittest.TestCase):

    def setUp(self):
        # is executed first
        X, y = load_wine(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.33, random_state=42)

    def test_accuracy_data(self):

        clf = RSLVQ(sigma=0.5, prototypes_per_class=5, batch_size=10)

        clf.partial_fit(self.X_train, self.y_train)

        y_pred = clf.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        kappa = cohen_kappa_score(self.y_test, y_pred)

        self.assertTrue(acc >= 0.745, msg='Accuracy has to be greater than 0.745, \
                        but was {}'.format(acc))
        self.assertTrue(kappa >= 0.620, msg='Kappa has to be greater than 0.620, \
                        but was {}'.format(kappa))

    def test_accuracy_stream(self):

        stream = SEAGenerator(random_state=42)
        stream.prepare_for_use()

        clf = RSLVQ(sigma=0.5, prototypes_per_class=2, batch_size=5)

        evaluator = EvaluatePrequential(show_plot=False, max_samples=20000,
                                        batch_size=5)

        evaluator.evaluate(stream, clf, model_names=['RSLVQ'])

        measurements = np.asarray(evaluator.get_measurements()[0])[0]

        self.assertTrue(measurements.get_accuracy() >= 0.85,
                        msg='Accuracy was {} but has to be greater than 0.85'.
                        format(measurements.get_accuracy()))
        self.assertTrue(measurements.get_kappa() >= 0.7,
                        msg='Kappa was {} but has to be greater than 0.7'.
                        format(measurements.get_kappa()))

    def test_sigma(self):
        with self.assertRaises(ValueError):
            RSLVQ(sigma=-1)

        RSLVQ(sigma=1)

    def test_protoypes(self):
        with self.assertRaises(ValueError):
            RSLVQ(prototypes_per_class=0)

        RSLVQ(prototypes_per_class=10)

    def test_batch_size(self):
        with self.assertRaises(ValueError):
            RSLVQ(batch_size=0)

        RSLVQ(batch_size=5)

    def test_epochs(self):
        with self.assertRaises(ValueError):
            RSLVQ(n_epochs=0)

        RSLVQ(n_epochs=2)


class TestARSLVQ(unittest.TestCase):

    def setUp(self):
        # is executed first
        X, y = load_wine(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.33, random_state=42)

    def test_accuracy_data(self):

        clf = ARSLVQ(sigma=0.5, prototypes_per_class=5, batch_size=10)

        clf.partial_fit(self.X_train, self.y_train)

        y_pred = clf.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        kappa = cohen_kappa_score(self.y_test, y_pred)

        self.assertTrue(acc >= 0.745, msg='Accuracy has to be greater than 0.745, \
                        but was {}'.format(acc))
        self.assertTrue(kappa >= 0.620, msg='Kappa has to be greater than 0.620, \
                        but was {}'.format(kappa))

    def test_accuracy_stream(self):

        stream = SEAGenerator(random_state=42)
        stream.prepare_for_use()

        clf = ARSLVQ(sigma=0.5, prototypes_per_class=2,
                     batch_size=5, decay_rate=0.999)

        evaluator = EvaluatePrequential(show_plot=False, max_samples=20000,
                                        batch_size=5)

        evaluator.evaluate(stream, clf, model_names=['ARSLVQ'])

        measurements = np.asarray(evaluator.get_measurements()[0])[0]

        self.assertTrue(measurements.get_accuracy() >= 0.85,
                        msg='Accuracy was {} but has to be greater than 0.85'.
                        format(measurements.get_accuracy()))
        self.assertTrue(measurements.get_kappa() >= 0.7,
                        msg='Kappa was {} but has to be greater than 0.7'.
                        format(measurements.get_kappa()))

    def test_sigma(self):
        with self.assertRaises(ValueError):
            ARSLVQ(sigma=-1)

        ARSLVQ(sigma=1)

    def test_protoypes(self):
        with self.assertRaises(ValueError):
            ARSLVQ(prototypes_per_class=0)

        ARSLVQ(prototypes_per_class=10)

    def test_batch_size(self):
        with self.assertRaises(ValueError):
            ARSLVQ(batch_size=0)

        ARSLVQ(batch_size=5)

    def test_decay_rate(self):
        with self.assertRaises(ValueError):
            ARSLVQ(decay_rate=0.0)
        with self.assertRaises(ValueError):
            ARSLVQ(decay_rate=1.0)

        ARSLVQ(decay_rate=0.999)


if __name__ == '__main__':
    unittest.main()
