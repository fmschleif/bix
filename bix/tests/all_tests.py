#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:42:08 2018

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""
import unittest
  
def run_all_tests():
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='', pattern='*.py')
    return unittest.TextTestRunner().run(suite)
     
if __name__ == '__main__':
    runner = run_all_tests()