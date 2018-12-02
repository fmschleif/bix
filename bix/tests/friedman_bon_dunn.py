#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 11:18:24 2018

@author: Christoph Raab <christophraab@outlook.de>
"""


from bix.utils.statistical_test_lib import friedman_test as Friedman, bonferroni_dunn_test as Bon_dunn
import pandas as pd
import unittest
import os

class TESTFRIEDDUNN(unittest.TestCase):
    
    def setUp(self):
        pwd = os.path.dirname(os.path.abspath(__file__))
        test_data_file = os.path.join(pwd, "friedman_bon_dunn_data.csv")
        self.accuracys_per_algorithm = \
            pd.read_csv(test_data_file, header=None).values.T
    
    def test_friedman(self):  
        f_value, p_value, rankings, pivots, chi = Friedman(*self.accuracys_per_algorithm)
        self.assertEqual(chi, 13.259)
        self.assertEqual(p_value, 0.004956048030304472)
        
    def test_bon_dunn(self):
        f_value, p_value, rankings, pivots, chi = Friedman(*self.accuracys_per_algorithm)
        algorithm_names = [str(i) for i in range(len(pivots))]
        pivot_dict = dict(zip(algorithm_names, pivots))
        comparisons, z_values, p_values, adj_p_values = Bon_dunn(ranks=pivot_dict)
        self.assertEqual(z_values[0], 3.252691193458119)
        self.assertEqual(len(z_values), 4)
        self.assertEqual(p_values[3], 0.43667663367489107)
        self.assertEqual(len(p_values), 4)
 
if __name__ == '__main__':
    unittest.main()
    