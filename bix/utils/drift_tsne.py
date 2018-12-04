# Starting point
from __future__ import division

import math
import sys
from random import random as rnd

import numpy as np
import pandas as pd

from bix.detectors.kswin import KSWIN
from skmultiflow.drift_detection.adwin import ADWIN

class DriftTSNE():

    
    def __init__(self, drift_handling, confidence):
        self.drift_handling = drift_handling
        self.confidence = confidence
        self.init_drift_detection = True
        self.drift_detected = False
        # KSWIN 1/1000 sonst 0.1
        
    def concept_drift_detection(self, X, Y):
        if self.init_drift_detection:
            if self.drift_handling == 'KS':
                self.cdd = [KSWIN(alpha=self.confidence) for elem in X.T]
            if self.drift_handling == 'ADWIN':
                self.cdd = [ADWIN(delta=self.confidence) for elem in X.T]
        self.init_drift_detection = False
        self.drift_detected = False
        
        if not self.init_drift_detection: 
            for elem,detector in zip(X.T, self.cdd):
                for e in elem:
                    detector.add_element(e)
                    if detector.detected_change():
                        self.drift_detected = True
        
        
        return self.drift_detected