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
#base_dir = '/data01/data01/01-projects/07-NOAA-CoastalAct/04-working/02-hurricane/hurrican_web_plot/'

#name = 'IRENE'
#year = '2011'

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

obs_xtra_days = datetime.timedelta(2)


remove_mean_diff = False

apply_bbox_bias = True
#San_area2
bias_bbox = [  -75.9 ,  38.5 ,  -73.3 , 41.3 ]
bias_calc_start =  datetime.datetime(2012, 10, 22, 11, 0)
bias_calc_end   =  datetime.datetime(2012, 10, 29, 23, 0)


if apply_bbox_bias and remove_mean_diff:
    sys.exit('ERROR: only one can be True ... ')




