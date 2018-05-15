from __future__ import division,print_function


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Functions for handling observations from ndbc and coops


"""

__author__ = "Saeed Moghimi"
__copyright__ = "Copyright 2018, UCAR/NOAA"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "moghimis@gmail.com"



import pandas    as pd
import numpy as np
import sys,os
import datetime


base_dir = '/disks/NASARCHIVE/saeed_moghimi/post/folium/coastal_act/'



name = 'SANDY'
year = '2012'

#year    = '2017'
#name    = 'IRMA'

#year    = '2016'
#name    = 'MATTHEW'

#year    = '2008'
#name    = 'IKE'


plot_cones = True
plot_sat   = False
remove_mean_diff = True
obs_xtra_days = datetime.timedelta(5)









