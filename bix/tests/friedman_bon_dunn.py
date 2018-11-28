#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 11:18:24 2018

@author: Christoph Raab <christophraab@outlook.de>
"""


from lib import friedman_test as Friedman, bonferroni_dunn_test as Bon_dunn
import datetime
import pandas as pd
import unittest


class TESTFRIEDDUNN(unittest.TestCase):
    
    def test_friedman(self):
        accuracys_per_algorithm = pd.read_csv("test_2_data.csv",header=None).values.T
        f_value, p_value, rankings, pivots,chi = Friedman(*accuracys_per_algorithm)
        self.assertEqual(chi,13.259)

if __name__ == '__main__':
    unittest.main()