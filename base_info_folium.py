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
from hurricane_funcs import get_nhc_storm_info


#base_dirf = '/disks/NASARCHIVE/saeed_moghimi/post/folium/coastal_act/wrk_dir/'
#base_dirf = '/data01/data01/01-projects/07-Maryland/02-working/02-hurricane/hurrican_web_plot_v04/'
base_dirf = '/mnt/c/Users/Saeed.Moghimi/Documents/work/linux_working/00-working/08-stofs-post/hurrican_web_plot/'
storms = defaultdict(dict)


if True: 
    key  = 'DORIAN'
    storms[key]['name' ]   = key
    storms[key]['year' ]   = '2019'
    storms[key]['start']   = datetime.datetime(2019, 8, 25)
    storms[key]['end'  ]   = datetime.datetime(2019, 9, 8)
    storms[key]['bbox' ]   = None
    #storms[key]['bbox' ]   = [-84.40, 9.90, -16.40, 38.20]


if False: 
    # 2020
    for nam in ['Hanna', 'Isaias', 'Laura', 'Marco', 'Beta', 'Alpha', 'Nana', 'Paulette', 'Sally', 'Teddy', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Iota']:
        storms[nam]['name' ]   = nam.upper()
        storms[nam]['year' ]   = '2020'
        storms[nam]['bbox' ]   = None

if False: 
    # 2019 storms
    _,_,df = get_nhc_storm_info ('2019','DORIAN')        

    ind = df.index
    for ii in range(len(ind)-1):
        nam = str(ind[ii+1])
        print ('  >  ',nam)
        storms[nam]['name' ]   = nam.upper()
        storms[nam]['year' ]   = '2019'
        storms[nam]['bbox' ]   = None


    key  = 'FLORENCE'
    storms[key]['name' ]   = key
    storms[key]['year' ]   = '2018'
    storms[key]['start']   = datetime.datetime(2018, 8, 26)
    storms[key]['end'  ]   = datetime.datetime(2018, 9, 26)
    storms[key]['bbox' ]   = [-84.40, 9.90, -16.40, 38.20]


    if True:    
        key  = 'IRENE'
        storms[key]['name' ]   = key
        storms[key]['year' ]   = '2011'
        storms[key]['start']   = None 
        storms[key]['end'  ]   = None
        storms[key]['bbox' ]   =  -82.50, 12.90, -56.50, 47.30  

    if True:    
        key  = 'SANDY'
        storms[key]['name' ]   = key
        storms[key]['year' ]   = '2012'
        storms[key]['start']   = None 
        storms[key]['end'  ]   = None
        storms[key]['bbox' ]   = -82.50, 10.50, -65.50, 41.80

    if True:    
        key  = 'ISAAC'
        storms[key]['name' ]   = key
        storms[key]['year' ]   = '2012'
        storms[key]['start']   = None 
        storms[key]['end'  ]   = None
        storms[key]['bbox' ]   = None

    if True:    
        key  = 'ARTHUR'
        storms[key]['name' ]   = key
        storms[key]['year' ]   = '2014'
        storms[key]['start']   = None 
        storms[key]['end'  ]   = None
        storms[key]['bbox' ]   = -82.0, 25.50, -63.50, 47.00

    if True:    
        key  = 'MATTHEW'
        storms[key]['name' ]   = key
        storms[key]['year' ]   = '2016'
        storms[key]['start']   = None 
        storms[key]['end'  ]   = None
        storms[key]['bbox' ]   = None

    if True:    
        key  = 'HERMINE'
        storms[key]['name' ]   = key
        storms[key]['year' ]   = '2016'
        storms[key]['start']   = None 
        storms[key]['end'  ]   = None
        storms[key]['bbox' ]   = None
     
    if True:    
        key  = 'IRMA'
        storms[key]['name' ]   = key
        storms[key]['year' ]   = '2017'
        storms[key]['start']   = None 
        storms[key]['end'  ]   = None
        storms[key]['bbox' ]   = None    

    if True:    
        key  = 'MARIA'
        storms[key]['name' ]   = key
        storms[key]['year' ]   = '2017'
        storms[key]['start']   = None 
        storms[key]['end'  ]   = None
        storms[key]['bbox' ]   = -82.50,9.90, -56.50, 44.00

    if True:    
        key  = 'HARVEY'
        storms[key]['name' ]   = key
        storms[key]['year' ]   = '2017'
        storms[key]['start']   = None 
        storms[key]['end'  ]   = None
        storms[key]['bbox' ]   = -99.70, 17.90, -88.10, 33.70


    if True:    
        key  = 'MICHAEL'
        storms[key]['name' ]   = key
        storms[key]['year' ]   = '2018'
        storms[key]['start']   = None 
        storms[key]['end'  ]   = None
        storms[key]['bbox' ]   = None

    if True:    
        key  = 'BARRY'
        storms[key]['name' ]   = key
        storms[key]['year' ]   = '2019'
        storms[key]['start']   = None 
        storms[key]['end'  ]   = None
        storms[key]['bbox' ]   = None

if False:    

    key  = 'KATRINA'
    storms[key]['name' ]   = key
    storms[key]['year' ]   = '2005'
    storms[key]['start']   = datetime.datetime(2005, 8, 20)
    storms[key]['end'  ]   = datetime.datetime(2005, 9, 3)
    storms[key]['bbox' ]   =  -97.35 , 17.78 , -66.10 , 33.66  

    key  = 'IKE'
    storms[key]['name' ]   = key
    storms[key]['year' ]   = '2008'
    storms[key]['start']   = None 
    storms[key]['end'  ]   = None
    storms[key]['bbox' ]   = None



## 2020
# Hanna and Isaias
# Laura and Marco
# Beta Alpha  Nana
# Paulette, Sally, and Teddy.
# Gamma   Delta  Epsilon  Zeta Eta Iota 


get_cops_wlev = True
get_cops_wind = True
get_ndbc_wave = True
get_ndbc_wind = True
get_usgs_hwm  = True

plot_cones = True
plot_sat   = False

obs_xtra_days = datetime.timedelta(1)


remove_mean_diff = True
apply_bbox_bias  = False
#San_area2
bias_bbox = [  -75.9 ,  38.5 ,  -73.3 , 41.3 ]
bias_calc_start =  datetime.datetime(2012, 10, 22, 11, 0)
bias_calc_end   =  datetime.datetime(2012, 10, 29, 23, 0)


if apply_bbox_bias and remove_mean_diff:
    sys.exit('ERROR: only one can be True ... ')




