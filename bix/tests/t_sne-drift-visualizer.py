#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:42:46 2018

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

import unittest
from bix.visualize.stream_t_sne import T_sne_stream_visualizer
from skmultiflow.data.sea_generator import SEAGenerator

class TestTSNEVisualizer(unittest.TestCase):
    
    def test_wrong_stream(self):
        with self.assertRaises(TypeError):
            T_sne_stream_visualizer(stream='tesla')
            
    def test_wrong_drift_handler(self):
        with self.assertRaises(TypeError):
            T_sne_stream_visualizer(stream=SEAGenerator(),
                                    drift_handling='TSWIN')
        
    def test_correct_init(self):
        T_sne_stream_visualizer(stream=SEAGenerator(), 
                                normalize=True,
                                path=None,
                                drift_handling='KS',
                                confidence=0.0001)
    
if __name__ == '__main__':
    unittest.main()