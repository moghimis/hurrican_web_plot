from __future__ import division,print_function


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Read obs from NDBC and Co-opc

# For IKE and Isable pts files in al092008_5day_052.zip are available




"""

__author__ = "Saeed Moghimi"
__copyright__ = "Copyright 2018, UCAR/NOAA"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "moghimis@gmail.com"
#
import pandas as pd
import numpy as np
#
from bs4 import BeautifulSoup
import requests
import lxml.html
import sys,os
#
from pyoos.collectors.ndbc.ndbc_sos import NdbcSos
from pyoos.collectors.coops.coops_sos import CoopsSos
from retrying import retry
import datetime
import cf_units
from io import BytesIO
from ioos_tools.ioos import collector2table
import pickle 
#
import geopandas as gpd
#####################################################
try:
    os.system('rm __pycache__/hurricane_funcs*'  )
    os.system('rm hurricane_funcs.pyc'  )

except:
    pass
if 'hurricane_funcs' in sys.modules:  
    del(sys.modules["hurricane_funcs"])
from hurricane_funcs import *
####################################################
try:
    os.system('rm __pycache__/base_info*.pyc'  )
    os.system('rm base_info*.pyc'  )
except:
    pass
if 'base_info' in sys.modules:  
    del(sys.modules["base_info"])
from base_info import *


if 'base_info_folium' in sys.modules:  
    del(sys.modules["base_info_folium"])
from base_info_folium import *


print('\n\n\n\n\n\n********************************************************')
print(            '*****  Storm name ',name, '      Year ',  year, '    *********')
print(            '******************************************************** \n\n\n\n\n\n')


#
import arrow
#       
if False:
    # not needed. will take from the storm specific obs list from coops and ndbc
    obs_station_list_gen()
#
#######

if int (year) > 2008:
    #for storms after 2010 works
    code,hurricane_gis_files = get_nhc_storm_info (year,name)
    base                     = download_nhc_gis_best_track(year,code)
    line,points,radii = read_gis_best_track(base,code)
    download_nhc_best_track(year,code)
    #
    start_txt = str (np.array( points.DTG)[ 0])
    end_txt   = str (np.array( points.DTG)[-1])
else:
    #for IKE
    #read file info
    code,hurricane_gis_files = get_nhc_storm_info (year,name)
    #donload gis zip files
    base = download_nhc_gis_files(hurricane_gis_files)
    # get advisory cones and track points
    cones,po,points = read_advisory_cones_info(hurricane_gis_files,base,year,code)
    start    = po[0] ['ADVDATE']
    end      = po[-1]['ADVDATE']
    
    start_txt = ('20' + start[:-2]).replace('/','')
    end_txt   = ('20' + end  [:-2]).replace('/','')

    print('\n\n\n\n  >>>>> Download and read all GIS data for Storm >',name, '      Year > ', year, '\n     **  This is an old STORM !!!!!! \n\n\n\n')

#
bounds = np.array(points.buffer(2).bounds)
lons   = np.r_[bounds[:,0],bounds[:,2]]
lats   = np.r_[bounds[:,1],bounds[:,3]]
#
bbox = lons.min(), lats.min(), lons.max(), lats.max()
#
start_dt = arrow.get(start_txt, 'YYYYMMDDhh').datetime - obs_xtra_days
end_dt   = arrow.get(end_txt  , 'YYYYMMDDhh').datetime + obs_xtra_days
#
# Note that the bounding box is derived from the track and the latest prediction cone.
strbbox = ', '.join(format(v, '.2f') for v in bbox)
print('\n\n\n\n\n\n********************************************************')
print(            '*****  Storm name ',name, '      Year ',  year, '    *********')
print('bbox: {}\nstart: {}\n  end: {}'.format(strbbox, start_dt, end_dt))
print(            '******************************************************** \n\n\n\n\n\n')
#
#########
# out dir
obs_dir = os.path.join(base_dirf,'obs')
#
#######
print('  > Get water level information  CO-OPS')
ssh, ssh_table = get_coops(
    start=start_dt,
    end=end_dt,
    sos_name='water_surface_height_above_reference_datum',
    units=cf_units.Unit('meters'),
    datum = 'MSL',
    bbox=bbox,
    )

write_csv(obs_dir, name, year, table=ssh_table    , data= ssh     , label='coops_ssh' )

#######
print('  > Get wind information CO-OPS')
wnd_obs, wnd_obs_table = get_coops(
    start=start_dt,
    end=end_dt,
    sos_name='wind_speed',
    units=cf_units.Unit('m/s'),
    bbox=bbox,
    )

write_csv(obs_dir, name, year, table=wnd_obs_table, data= wnd_obs , label='coops_wind')

######
print('  > Get wind ocean information (ndbc)')
wnd_ocn, wnd_ocn_table = get_ndbc(
    start=start_dt,
    end=end_dt,
    sos_name='winds',
    bbox=bbox,
    )

write_csv(obs_dir, name, year, table=wnd_ocn_table, data= wnd_ocn , label='ndbc_wind' )

######
print('  > Get wave ocean information (ndbc)')
wav_ocn, wav_ocn_table = get_ndbc(
    start=start_dt,
    end=end_dt,
    sos_name='waves',
    bbox=bbox,
    )

write_csv(obs_dir, name, year, table=wav_ocn_table, data= wav_ocn , label='ndbc_wave' )
######

if False:
    # test reading files
    ssh_table1    , ssh1      = read_csv (obs_dir, name, year, label='coops_ssh' )
    wnd_obs_table1, wnd_obs1  = read_csv (obs_dir, name, year, label='coops_wind')
    wnd_ocn_table1, wnd_ocn1  = read_csv (obs_dir, name, year, label='ndbc_wind' )
    wav_ocn_table1, wav_ocn1 = read_csv (obs_dir, name, year, label='ndbc_wave' )


#
# back up script file
args=sys.argv
scr_name = args[0]
scr_dir = os.path.join(obs_dir, name+year)
os.system('cp -fr ' + scr_name + '    ' + scr_dir)
#
#with open(pick, "rb") as f:
#    w = pickle.load(f)

#f = open(pick, "rb")
#w = pickle.load(f)


#if __name__ == "__main__":
#    main()















