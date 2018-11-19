#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 08:26:44 2018

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

import unittest
from bix.preprocessing.CrossValidation import CrossValidation
from bix.classifiers.batch_adaptive_rslvq import RSLVQ
from skmultiflow.data.agrawal_generator import AGRAWALGenerator

clfs = [RSLVQ(gradient_descent='Adadelta'),
        RSLVQ()]

streams = [AGRAWALGenerator()]

cv = CrossValidation(clfs=clfs, streams=streams)

cv.test()
cv.save_summary()