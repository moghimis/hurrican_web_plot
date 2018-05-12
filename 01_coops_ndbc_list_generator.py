from __future__ import division,print_function


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Read obs from NDBC and Co-opc


"""

__author__ = "Saeed Moghimi"
__copyright__ = "Copyright 2018, UCAR/NOAA"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "moghimis@gmail.com"



import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
import requests
import lxml.html
import sys,os


try:
    os.system('rm __pycache__/hurricane_funcs*'  )
except:
    pass
if 'hurricane_funcs' in sys.modules:  
    del(sys.modules["hurricane_funcs"])
from hurricane_funcs import *





