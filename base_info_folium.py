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


#base_dirf = '/disks/NASARCHIVE/saeed_moghimi/post/folium/coastal_act/'
base_dirf = '/data01/data01/01-projects/07-Maryland/02-working/02-hurricane/hurrican_web_plot/'

#
#year    = '2008'
#name    = 'IKE'

#no GIS data
#year    = '2003'
#name    = 'ISABEL'

#year    = '2017'
#name    = 'IRMA'

#name = 'SANDY'
#year = '2012'

#name = 'IRENE'
#year = '2011'

#name = 'HARVEY'
#year = '2017'

#name = 'MARIA'
#year = '2017'

#name = 'MATTHEW'
#year = '2016'

#name = 'ISAAC'
#year = '2012'

#name = 'HERMINE'
#year = '2016'

#name = 'ARTHUR'
#year = '2014'





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




