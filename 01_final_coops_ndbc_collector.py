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
import dateparser

import cf_units
from io import BytesIO
from ioos_tools.ioos import collector2table
import pickle 
#

sys.path.append('/disks/NASARCHIVE/saeed_moghimi/opt/python-packages/')

import geopandas as gpd

from shapely.geometry import LineString
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
    os.system('rm __pycache__/base_info_folium*.pyc'  )
    os.system('rm base_info_folium*.pyc'  )
except:
    pass


if 'base_info' in sys.modules:  
    del(sys.modules["base_info"])
#from base_info import *


if 'base_info_folium' in sys.modules:  
    del(sys.modules["base_info_folium"])
from base_info_folium import *

#
import arrow
#       
if False:
    # not needed. will take from the storm specific obs list from coops and ndbc
    obs_station_list_gen()
#
#######
# out dir
obs_dir = os.path.join(base_dirf,'obs')

   
if get_usgs_hwm:
    for key in storms.keys():
        name = storms[key]['name']
        year = storms[key]['year']
        print('  > Get USGS HWM for ', name)
        try:
            write_high_water_marks(obs_dir, name, year)  
        except:
            print (' > Get USGS HWM for ', name , '   ERROR ...')


for key in storms.keys():
    name = storms[key]['name']
    year = storms[key]['year']

    print('\n\n\n\n\n\n********************************************************')
    print(            '*****  Storm name ',name, '      Year ',  year, '    *********')
    print(            '******************************************************** \n\n\n\n\n\n')



    
    
    #if bbox_from_best_track:
    try:    
        
        #bbox_from_best_track = False
        code,hurricane_gis_files,df = get_nhc_storm_info (year,name)        
        
        ###############################################################################
        #download gis zip files
        base = download_nhc_gis_files(hurricane_gis_files)
        # get advisory cones and track points
        cones,pts_actual,points_actual = read_advisory_cones_info(hurricane_gis_files,base,year,code)
        start    = pts_actual[0] ['FLDATELBL']
        end      = pts_actual[-1]['FLDATELBL']
        #start_txt_actual = ('20' + start[:-2]).replace('/','')
        #end_txt_actual   = ('20' + end  [:-2]).replace('/','')


        #print('\n\n\n\n\n\n ********************************************************')
        #for key1 in pts_actual[0].keys():
        #    print(            '*****  pts_actual[0] [', key1, ']',pts_actual[0] [key1]   ,  '*********')
        #print(            '******************************************************** \n\n\n\n\n\n')

        start_dt = dateparser.parse(start,settings={"TO_TIMEZONE": "UTC"}).replace(tzinfo=None) - obs_xtra_days
        end_dt   = dateparser.parse(end  ,settings={"TO_TIMEZONE": "UTC"}).replace(tzinfo=None) + obs_xtra_days   
        
        #try:
        #    # bbox_from_best_track:
        #    start_txt = start_txt_best
        #    end_txt   = end_txt_best
        #    #bbox      = bbox_best
        #except:
        #    start_txt = start_txt_actual
        #    end_txt   = end_txt_actual

        #
        #start_dt = arrow.get(start_txt, 'YYYYMMDDhh').datetime - obs_xtra_days
        #end_dt   = arrow.get(end_txt  , 'YYYYMMDDhh').datetime + obs_xtra_days    

        
        #if False:
        # get bbox from actual data
        last_cone = cones[-1]['geometry'].iloc[0]
        track = LineString([point['geometry'] for point in pts_actual])
        lons_actual = track.coords.xy[0]
        lats_actual = track.coords.xy[1]
        bbox_actual = min(lons_actual)-2, min(lats_actual)-2, max(lons_actual)+2, max(lats_actual)+2
        ################################################################################

        # Find the bounding box to search the data.
        bbox_from_best_track = False
        bbox      = bbox_actual
    except:
        start_dt   = storms[key]['start']
        end_dt     = storms[key]['end'  ]
        bounds  = storms[key]['bbox' ]


    if storms[key]['bbox'] is not None:
        bbox = storms[key]['bbox']
    
    #print('\n\n\n\n  >>>>> Download and read all GIS data for Storm >',name, '      Year > ', year, '\n     **  This is an old STORM !!!!!! \n\n\n\n')

    #
    # Note that the bounding box is derived from the track and the latest prediction cone.
    strbbox = ', '.join(format(v, '.2f') for v in bbox)


    #
    # Note that the bounding box is derived from the track and the latest prediction cone.
    strbbox = ', '.join(format(v, '.2f') for v in bbox)
    print('\n\n\n\n\n\n********************************************************')
    print(            '*****  Storm name ',name, '      Year ',  year, '    *********')
    print('bbox: {}\nstart: {}\n  end: {}'.format(strbbox, start_dt, end_dt))
    print(            '******************************************************** \n\n\n\n\n\n')
    #
    #########
    
    if get_cops_wlev:
        try:
            print('  > Get water level information CO-OPS ... ')
            
            # ["MLLW","MSL","MHW","STND","IGLD", "NAVD"]
            datum =  'NAVD'
            datum =  'MSL'
            print ('datum=', datum )
            ssh, ssh_table = get_coops(
                start=start_dt,
                end=end_dt,
                sos_name='water_surface_height_above_reference_datum',
                units=cf_units.Unit('meters'),
                datum = datum ,
                bbox=bbox,
                )

            write_csv(obs_dir, name, year, table=ssh_table    , data= ssh     , label='coops_ssh' )

        except:
            print('  > Get water level information  CO-OPS  >>>> ERRORRRRR')
    ######
    
    

    if get_cops_wind:
        try:
            print('  > Get wind information CO-OPS ... ')
            wnd_obs, wnd_obs_table = get_coops(
                start=start_dt,
                end=end_dt,
                sos_name='wind_speed',
                units=cf_units.Unit('m/s'),
                bbox=bbox,
                )

            write_csv(obs_dir, name, year, table=wnd_obs_table, data= wnd_obs , label='coops_wind')
        except:
            print('  > Get wind information CO-OPS >>> ERORRRR')
    ######
    if get_ndbc_wind:
        try:
            print('  > Get wind ocean information (ndbc) ... ')
            wnd_ocn, wnd_ocn_table = get_ndbc(
                start=start_dt,
                end=end_dt,
                sos_name='winds',
                bbox=bbox,
                )

            write_csv(obs_dir, name, year, table=wnd_ocn_table, data= wnd_ocn , label='ndbc_wind' )
        except:
            print('  > Get wind ocean information (ndbc)  >>> ERRRORRRR')
    ######
    if get_ndbc_wave:
        try:
            print('  > Get wave ocean information (ndbc) ... ')
            wav_ocn, wav_ocn_table = get_ndbc(
                start=start_dt,
                end=end_dt,
                sos_name='waves',
                bbox=bbox,
                )

            write_csv(obs_dir, name, year, table=wav_ocn_table, data= wav_ocn , label='ndbc_wave' )
        except:  
            print('  > Get wave ocean information (ndbc)  >>> ERRORRRR ')
    ######

    



if False:
    # test reading files
    ssh_table1    , ssh1      = read_csv (obs_dir, name, year, label='coops_ssh' )
    wnd_obs_table1, wnd_obs1  = read_csv (obs_dir, name, year, label='coops_wind')
    wnd_ocn_table1, wnd_ocn1  = read_csv (obs_dir, name, year, label='ndbc_wind' )
    wav_ocn_table1, wav_ocn1 = read_csv (obs_dir, name, year, label='ndbc_wave' )

    #if False:
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















