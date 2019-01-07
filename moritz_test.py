#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 10:45:06 2019

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

from bix.classifiers.rslvq import RSLVQ
from bix.classifiers.rrslvq import RRSLVQ
from skmultiflow.data.led_generator_drift import LEDGeneratorDrift
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential

led_a = ConceptDriftStream(stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                            drift_stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=7),
                            random_state=None,
                            alpha=90.0,
                            position=250000)

led_a.prepare_for_use()

clf = RRSLVQ()

eval = EvaluatePrequential(batch_size=10)
eval.evaluate(stream=led_a, model=clf)
