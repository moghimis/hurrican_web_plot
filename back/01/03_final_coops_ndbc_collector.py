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

from pyoos.collectors.ndbc.ndbc_sos import NdbcSos
from pyoos.collectors.coops.coops_sos import CoopsSos
from retrying import retry
import datetime
import cf_units
from io import BytesIO
from ioos_tools.ioos import collector2table
import pickle 

import geopandas as gpd

try:
    os.system('rm __pycache__/hurricane_funcs*'  )
except:
    pass
if 'hurricane_funcs' in sys.modules:  
    del(sys.modules["hurricane_funcs"])
from hurricane_funcs import *

import arrow



def main():
    #Sandy
    name = 'SANDY'
    year = '2012'
    obs_xtra_days = datetime.timedelta(5)
    
    base_dir = 'obs/'
    ########
    ########

    code,hurricane_gis_files = get_nhc_storm_info (year,name)
    base                     = download_nhc_gis_best_track(year,code)
    line,points,radii = read_gis_best_track(base,code)
    #
    bounds = np.array(points.buffer(2).bounds)
    lons = np.r_[bounds[:,0],bounds[:,2]]
    lats = np.r_[bounds[:,1],bounds[:,3]]
    #
    bbox = lons.min(), lats.min(), lons.max(), lats.max()
    #
    start_txt = str (np.array( points.DTG)[ 0])
    end_txt   = str (np.array( points.DTG)[-1])
    #
    start_dt = arrow.get(start_txt, 'YYYYMMDDhh').datetime - obs_xtra_days
    end_dt   = arrow.get(end_txt  , 'YYYYMMDDhh').datetime + obs_xtra_days
    #
    # Note that the bounding box is derived from the track and the latest prediction cone.
    strbbox = ', '.join(format(v, '.2f') for v in bbox)
    print('bbox: {}\nstart: {}\n  end: {}'.format(strbbox, start_dt, end_dt))
    
    ######
    print('  > Get water level information  CO-OPS')
    ssh, ssh_table = get_coops(
        start=start_dt,
        end=end_dt,
        sos_name='water_surface_height_above_reference_datum',
        units=cf_units.Unit('meters'),
        datum = 'MSL',
        bbox=bbox,
    )
    
    ######
    print('  > Get wind information CO-OPS')
    wnd_obs, wnd_obs_table = get_coops(
        start=start_dt,
        end=end_dt,
        sos_name='wind_speed',
        units=cf_units.Unit('m/s'),
        bbox=bbox,
        )

    #####
    print('  > Get wind ocean information (ndbc)')
    wnd_ocn, wnd_ocn_table = get_ndbc(
        start=start_dt,
        end=end_dt,
        sos_name='winds',
        bbox=bbox,
        )
    
    #####
    print('  > Get wave ocean information (ndbc)')
    wav_ocn, wav_ocn_table = get_ndbc(
        start=start_dt,
        end=end_dt,
        sos_name='waves',
        bbox=bbox,
        )

    #####
    all_data = dict(wnd_ocn =wnd_ocn   , wnd_ocn_table = wnd_ocn_table,
                    wav_ocn = wav_ocn  , wav_ocn_table = wav_ocn_table,
                    ssh     =  ssh     , ssh_table     = ssh_table,
                    wnd_obs =  wnd_obs , wnd_obs_table = wnd_obs_table)

    #####
    bbox_txt = str(bbox).replace(' ','_').replace(',','_').replace('[','_').replace(']','_')
    scr_dir    = base_dir + '/' + name+year+'/'
    os.system('mkdir -p ' + scr_dir)

   
    pickname = scr_dir + name + year + bbox_txt.replace('(','_').replace(')','_') + '.pik2'
    f = open(pickname, 'wb')
    pickle.dump(all_data,f,protocol=2)
    f.close()
    
   
    pickname = scr_dir + name + year + bbox_txt.replace('(','_').replace(')','_') + '.pik3'
    f = open(pickname, 'wb')
    pickle.dump(all_data,f)
    f.close()




    
    #
    # back up script file
    args=sys.argv
    scr_name = args[0]
    os.system('cp -fr ' + scr_name + '    ' + scr_dir)
    #
    #with open(pick, "rb") as f:
    #    w = pickle.load(f)

    #f = open(pick, "rb")
    #w = pickle.load(f)


if __name__ == "__main__":
    main()















