__author__ = 'leo@opensignal.com'
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))


setup(name='PyUpSet',
      version='0.1.1',
      description='Python implementation of the UpSet visualisation suite by Lex et al.',
      author = 'Leonardo Baldassini',
      author_email= 'leo@opensignal.com',
      url='https://github.com/ImSoErgodic/py-upset',
      license='MIT',
      classifiers= [
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering :: Visualization',
      'Programming Language :: Python :: 3'],
      install_requires=['pandas', 'matplotlib', 'numpy'],
      package_data={'test_data_dictionary':['test_data_dict.pckl']})