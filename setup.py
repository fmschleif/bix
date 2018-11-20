#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 08:52:48 2018

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

import setuptools

setuptools.setup(name='bix',
      version='0.1.0',
      description='Business Intelligence Excellence',
      url='https://github.com/ChristophRaab/bix',
      author='Moritz Heusinger',
      author_email='moritz.heusinger@gmail.com',
      license='MIT',
      packages=setuptools.find_packages(),
      install_requires=[
          'joblib',
          'scikit-learn'
      ],
      zip_safe=False)