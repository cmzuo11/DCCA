#!/usr/bin/env python
"""
# Author: cmzuo
# Created Time : Jan 27 Oct 2021 02:42:37 PM CST
# File Name: setup.py
# Description:
"""
from setuptools import setup, find_packages

with open('used_package.txt') as f:
    requirements = f.read().splitlines()

setup(name='DCCA',
      version='1.0.1',
      packages=find_packages(),
      description='Deep cross-omics cycle attention (DCCA) model for joint analysis of single-cell multi-omics data',
      long_description='',


      author='Chunman Zuo',
      author_email='',
      url='https://github.com/cmzuo11/DCCA',
      scripts=['Main_SNARE_seq.py'],
      install_requires=requirements,
      python_requires='>3.6.12',
      license='MIT',

      classifiers=[
          'Development Status :: 1 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Bioinformatics',
     ],
     )