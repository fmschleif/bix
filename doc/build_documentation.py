#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 01:57:47 2018

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

import os, sys

if (len(sys.argv) == 1):
    os.system('sphinx-apidoc -o source .. && make clean || rd /s /q build || echo t>NUL && ' 
              'sphinx-build source build -a -b html')
elif (sys.argv[1] == '--help' or sys.argv[1] == '-h'):
    print('This script updates and rebuilds the documentation, just execute it without args')
else:
    print('Invaild argument, use --help or -h to show help\n' \
          'Execute without arguments to update and rebuild documentation')