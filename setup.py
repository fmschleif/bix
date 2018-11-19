#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 08:52:48 2018

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

from setuptools import setup

setup(name='bix',
      version='0.1',
      description='Business Intelligence Excellence',
      url='https://github.com/foxriver76/bix',
      author='Moritz Heusinger',
      author_email='moritz.heusinger@gmail.com',
      license='MIT',
      packages=['bix'],
      install_requires=[
          'joblib',
      ],
      zip_safe=False)