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
from   collections import defaultdict

#base_dirf = '/disks/NASARCHIVE/saeed_moghimi/post/folium/coastal_act/'
base_dirf = '/data01/data01/01-projects/07-Maryland/02-working/02-hurricane/hurrican_web_plot/'


storms = defaultdict(dict)

if True:
    key  = 'IRENE'
    storms[key]['name' ]   = key
    storms[key]['year' ]   = '2011'
    storms[key]['start']   = None 
    storms[key]['end'  ]   = None
    storms[key]['bbox' ]   =  -76.50, 12.90, -56.50, 47.30  

    key  = 'MARIA'
    storms[key]['name' ]   = key
    storms[key]['year' ]   = '2017'
    storms[key]['start']   = None 
    storms[key]['end'  ]   = None
    storms[key]['bbox' ]   = -82.50,9.90, -56.50, 44.00
    
    

if False:


    key  = 'MATTHEW'
    storms[key]['name' ]   = key
    storms[key]['year' ]   = '2016'
    storms[key]['start']   = None 
    storms[key]['end'  ]   = None
    storms[key]['bbox' ]   = None

    key  = 'ISAAC'
    storms[key]['name' ]   = key
    storms[key]['year' ]   = '2012'
    storms[key]['start']   = None 
    storms[key]['end'  ]   = None
    storms[key]['bbox' ]   = None

    key  = 'HARVEY'
    storms[key]['name' ]   = key
    storms[key]['year' ]   = '2017'
    storms[key]['start']   = None 
    storms[key]['end'  ]   = None
    storms[key]['bbox' ]   = -99.70, 17.90, -88.10, 33.70

    key  = 'IKE'
    storms[key]['name' ]   = key
    storms[key]['year' ]   = '2008'
    storms[key]['start']   = None 
    storms[key]['end'  ]   = None
    storms[key]['bbox' ]   = None

    key  = 'ARTHUR'
    storms[key]['name' ]   = key
    storms[key]['year' ]   = '2014'
    storms[key]['start']   = None 
    storms[key]['end'  ]   = None
    storms[key]['bbox' ]   = -82.0, 25.50, -63.50, 47.00

    key  = 'HERMINE'
    storms[key]['name' ]   = key
    storms[key]['year' ]   = '2016'
    storms[key]['start']   = None 
    storms[key]['end'  ]   = None
    storms[key]['bbox' ]   = None

    key  = 'IRMA'
    storms[key]['name' ]   = key
    storms[key]['year' ]   = '2017'
    storms[key]['start']   = None 
    storms[key]['end'  ]   = None
    storms[key]['bbox' ]   = None

    key  = 'SANDY'
    storms[key]['name' ]   = key
    storms[key]['year' ]   = '2012'
    storms[key]['start']   = None 
    storms[key]['end'  ]   = None
    storms[key]['bbox' ]   = -82.50, 10.50, -65.50, 41.80


#
#year    = '2008'
#name    = 'IKE'

#no GIS data
#year    = '2003'
#name    = 'ISABEL'

<<<<<<< HEAD
base_dirf = '/disks/NASARCHIVE/saeed_moghimi/post/folium/coastal_act/'
#base_dir = '/data01/data01/01-projects/07-NOAA-CoastalAct/04-working/02-hurricane/hurrican_web_plot/'

#name = 'IRENE'
#year = '2011'
=======
#year    = '2017'
#name    = 'IRMA'
>>>>>>> 412587def53030109142f5bc426f5fbda3f6a066

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

<<<<<<< HEAD
year    = '2017'
name    = 'IRMA'
=======
>>>>>>> 412587def53030109142f5bc426f5fbda3f6a066




plot_cones = False
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




