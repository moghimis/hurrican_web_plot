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
####################################################

import arrow
        
if False:
    # not needed. will take from the storm specific obs list from coops and ndbc
    obs_station_list_gen()



obs_xtra_days = datetime.timedelta(5)


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


bbox_txt   = str(bbox).replace(' ','_').replace(',','_').replace('[','_').replace(']','_')
scr_dir    = base_dir + '/' + name+year+'/'
pickname   = scr_dir + name + year + bbox_txt.replace('(','_').replace(')','_')


if  os.path.exists(pickname+'.pik2' or pickname+'.pik3'):
    print (' read from pickle')

    f = open(pickname+'.pik3', "rb")
    all_data = pickle.load(f)

    ssh             = all_data['ssh']
    ssh_table       = all_data['ssh_table']
    #
    wnd_obs         = all_data['wnd_obs']
    wnd_obs_table   = all_data['wnd_obs_table']
    #
    wav_ocn         = all_data['wav_ocn']
    wav_ocn_table   = all_data['wav_ocn_table']
    #
    wnd_ocn         = all_data['wnd_ocn']
    wnd_ocn_table   = all_data['wnd_ocn_table']

else:
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
    print('  > write pickle files')

    all_data = dict(wnd_ocn = wnd_ocn  , wnd_ocn_table = wnd_ocn_table,
                    wav_ocn = wav_ocn  , wav_ocn_table = wav_ocn_table,
                    ssh     = ssh      , ssh_table     = ssh_table,
                    wnd_obs = wnd_obs  , wnd_obs_table = wnd_obs_table)

    #####
    bbox_txt = str(bbox).replace(' ','_').replace(',','_').replace('[','_').replace(']','_')
    scr_dir    = base_dir + '/' + name+year+'/'
    pickname2 = pickname + '.pik2'
    
    os.system('mkdir -p ' + scr_dir)
    f = open(pickname2, 'wb')
    pickle.dump(all_data,f,protocol=2)
    f.close()


    pickname3 = pickname + '.pik3'
    f = open(pickname3, 'wb')
    pickle.dump(all_data,f)
    f.close()




def write_csv(base_dir, name, year, table, data, label):
    """
    examples
    print('  > write csv files')
    write_csv(base_dir, name, year, table=wnd_ocn_table, data= wnd_ocn , label='ndbc_wind' )
    write_csv(base_dir, name, year, table=wav_ocn_table, data= wav_ocn , label='ndbc_wave' )
    write_csv(base_dir, name, year, table=ssh_table    , data= ssh     , label='coops_ssh' )
    write_csv(base_dir, name, year, table=wnd_obs_table, data= wnd_obs , label='coops_wind')
    
    """
    label   = 'coops_ssh'
    out_dir =  os.path.join(base_dir,name+year) 
    table   = ssh_table
    data    = ssh

    outt    = os.path.join(base_dir, name+year,label)
    outd    = os.path.join(outt,'data')  
    if not os.path.exists(outd):
        os.makedirs(outd)

    table.to_csv(os.path.join(outt,'table.csv'))
    stations = table['station_code']

    for ista in range(len(stations)):
        sta   = stations [ista]
        fname = os.path.join(outd,sta+'.csv')
        data[ista].to_csv(fname)
        
        fmeta    = os.path.join(outd,sta)+'_metadata.csv'
        metadata = pd.DataFrame.from_dict( data[ista]._metadata , orient="index")
        metadata.to_csv(fmeta)
     

def read_csv(base_dir, name, year, label):
    """
    examples
    print('  > write csv files')
    write_csv(base_dir, name, year, table=wnd_ocn_table, data= wnd_ocn , label='ndbc_wind' )
    write_csv(base_dir, name, year, table=wav_ocn_table, data= wav_ocn , label='ndbc_wave' )
    write_csv(base_dir, name, year, table=ssh_table    , data= ssh     , label='coops_ssh' )
    write_csv(base_dir, name, year, table=wnd_obs_table, data= wnd_obs , label='coops_wind')
    
    """
    outt    = os.path.join(base_dir, name+year,label)
    outd    = os.path.join(outt,'data')  
    if not os.path.exists(outd):
       sys.exit('ERROR',outd )

    table2 = pd.read_csv(os.path.join(outt,'table.csv')).set_index('station_name')
    stations = table['station_code']

    data     = []
    metadata = []
    for ista in range(len(stations)):
        sta   = stations [ista]
        fname = os.path.join(outd,sta)+'.csv'
        data.append(pd.read_csv(fname))
    
        fmeta = os.path.join(outd,sta) + '_metadata.csv'
        metadata.append(pd.read_csv(fmeta))
        
    return table,data,metadata




print('  > write csv files')
write_csv(base_dir, name, year, table=ssh_table    , data= ssh     , label='coops_ssh' )
write_csv(base_dir, name, year, table=wnd_obs_table, data= wnd_obs , label='coops_wind')
write_csv(base_dir, name, year, table=wnd_ocn_table, data= wnd_ocn , label='ndbc_wind' )
write_csv(base_dir, name, year, table=wav_ocn_table, data= wav_ocn , label='ndbc_wave' )



ssh_table,ssh          = read_csv (base_dir, name, year, label='coops_ssh' )
wnd_obs_table,wnd_obs  = read_csv (base_dir, name, year, label='coops_wind')
wnd_ocn_table,wnd_ocn  = read_csv (base_dir, name, year, label='ndbc_wind' )
wav_ocn_table, wav_ocn, wav_ocn_meta  = read_csv (base_dir, name, year, label='ndbc_wave' )


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


#if __name__ == "__main__":
#    main()















