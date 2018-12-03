#!/bin/bash
sphinx-apidoc -o source/ ../
make clean
make html
