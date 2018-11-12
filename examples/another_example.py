#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 17:08:11 2018

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

from skmultiflow.data.sea_generator import SEAGenerator
from rslvq import RSLVQ
from adaptive_rslvqs_batch import RSLVQ as BARSLVQ
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from arslvq import RSLVQ as ARSLVQ
import numpy as np
from rslvq_stream import RSLVQ as MasterRSLVQ

stream = SEAGenerator()
stream.prepare_for_use()

clf = [RSLVQ(batch_size=5), 
       BARSLVQ(gradient_descent='Adadelta', batch_size=5, decay_rate=0.9, sigma=1.0),
       RSLVQ(),
       MasterRSLVQ(gradient_descent='Adadelta', decay_rate=0.999, sigma=1.0)]

evaluator = EvaluatePrequential(max_samples=100000, batch_size=5,
                                show_plot=True)
evaluator.evaluate(stream=stream, model=clf, model_names=['BRSLVQ', 'BARSLVQ', 'RSLVQ', 'MRSLVQ'])

measurements = np.asarray(evaluator.get_measurements()[0])[0]

