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

#from pyoos.collectors.ndbc.ndbc_sos import NdbcSos
#from pyoos.collectors.coops.coops_sos import CoopsSos
#from retrying import retry
#import datetime
#import cf_units
#from io import BytesIO
#from ioos_tools.ioos import collector2table
#import pickle 


###############3
## Other useful functions

def coops_list_datatypes():
    # TODO: Get from GetCaps
    return ["PreliminarySixMinute",
            "PreliminaryOneMinute",
            "VerifiedSixMinute",
            "VerifiedHourlyHeight",
            "VerifiedHighLow",
            "VerifiedDailyMean",
            "SixMinuteTidePredictions",
            "HourlyTidePredictions",
            "HighLowTidePredictions"]

def coops_list_datums():
    # TODO: Get from GetCaps
    return ["MLLW",
            "MSL",
            "MHW",
            "STND",
            "IGLD",
            "NAVD"]

def coops_list_observedPropertys():    
    return [
        "air_temperature",
        "air_pressure",
        "sea_water_electrical_conductivity",
        "sea_water_speed",
        "direction_of_sea_water_velocity",
        "sea_water_salinity",
        "water_surface_height_above_reference_datum",
        "sea_surface_height_amplitude_due_to_equilibrium_ocean_tide",
        "sea_water_temperature",
        "wind_from_direction",
        "wind_speed",
        "wind_speed_of_gust",
        "harmonic_constituents",
        "datums",
        "relative_humidity",
        "rain_fall",
        "visibility"]



def get_coops_stations_info(type = 'wlev'):
    """
    table coops meteo stations
    Meteorological & Ancillary Stations Via CO-OPS/SOS
    https://opendap.co-ops.nos.noaa.gov/ioos-dif-sos/ClientGetter?p=8
    
    table coops Water Level Stations Via CO-OPS/SOS
    https://opendap.co-ops.nos.noaa.gov/ioos-dif-sos/ClientGetter?p=6
    
    Active Current Meter Stations Via CO-OPS/SOS
    https://opendap.co-ops.nos.noaa.gov/ioos-dif-sos/ClientGetter?p=4
    
    """
    if   type == 'wlev': 
        num = '6'
    elif type == 'mete':
        num = '8'
    elif type == 'curr':
        num = '4'
    else:
        sys.exit(' ERORR: Not implemeted yet ..')          
        
    url  = 'https://opendap.co-ops.nos.noaa.gov/ioos-dif-sos/ClientGetter?p={}'.format(num)
    r    = requests.get(url)
    
    soup = BeautifulSoup(r.content, 'lxml')
    
    table = soup.findAll('table')[2]
    
    
    tab   = []
    names = []
    for row in table.find_all('td'):
        tmp = row.get_text().strip()
        tab.append(tmp)
    
    if type == 'curr':
        ncol = 6
    else:
        ncol = 5 
    tab = np.array(tab).reshape(len(tab)//ncol,ncol)
    
    if type == 'curr':
        tab = np.delete(tab,obj=3,axis=1)
    
    df = pd.DataFrame(
        data=tab[:],
        columns=['Station ID', 'Station Name', 'Deployed' , 'Latitude' , 'Longitude'],
    ).set_index('Station ID')
    
    
    
    for col in df.columns:
         try:
             df[col]  = df[col].astype('float64') 
         except:
             pass

    return df

def get_ndbc_stations_info(type = 'wave'):
    """
    https://sdf.ndbc.noaa.gov/stations.shtml
    
    Which station type
    type == 'wave':
    type == 'wind':
    type == 'wlev':    "Waterlevel"
    type == 'sst':     "Watertemperature"
    type == 'pres':     "Barometricpressure"

    """
    url  = 'https://sdf.ndbc.noaa.gov/stations.shtml'
    r    = requests.get(url)
    
    soup = BeautifulSoup(r.content, 'lxml')
    
    table = soup.findAll('table')[5]
    
    tab   = []
    names = []
    for row in table.find_all('td'):
        tmp = row.get_text().strip()
        tab.append(tmp)
    
    ncol = 6
    tab = np.array(tab).reshape(len(tab)//ncol,ncol)
   
    df = pd.DataFrame(
        data=tab[:],
        columns=['Station ID', 'Station Name','Owner', 'Latitude' , 'Longitude', 'Sensor'],
    ).set_index('Station ID')
    
    if   type == 'wave':
        df = df[df.Sensor.str.contains("Waves") == True]
    elif type == 'wind':
        df = df[df.Sensor.str.contains("Wind") == True]
    elif type == 'wlev':
        df = df[df.Sensor.str.contains("Waterlevel") == True]
    elif type == 'sst':
        df = df[df.Sensor.str.contains("Watertemperature") == True]
    elif type == 'pres':
        df = df[df.Sensor.str.contains("Barometricpressure") == True]
    else:
        sys.exit(' ERORR: Not implemeted yet ..')  


    for col in df.columns:
         try:
             df[col]  = df[col].astype('float64') 
         except:
             pass


    return df


def get_mask(bbox,lons,lats):
    mask     =    ~(( lons > bbox[0]) & 
                    ( lons < bbox[2]) & 
                    ( lats > bbox[1]) & 
                    ( lats < bbox[3]))
    
    return mask




coops_wlev_stations = get_coops_stations_info(type = 'wlev')
coops_mete_stations = get_coops_stations_info(type = 'mete')
ndbc_wave_stations  = get_ndbc_stations_info(type = 'wave')
ndbc_wind_stations  = get_ndbc_stations_info(type = 'wind')




#hsofs
bbox = [-99.0, 5.0, -52.8, 46.3]


coops_wlev_stations = coops_wlev_stations  [get_mask(bbox,coops_wlev_stations.Longitude,coops_wlev_stations.Latitude)]
coops_mete_stations = coops_mete_stations  [get_mask(bbox,coops_mete_stations.Longitude,coops_mete_stations.Latitude)]
ndbc_wave_stations  = ndbc_wave_stations   [get_mask(bbox,ndbc_wave_stations .Longitude,ndbc_wave_stations .Latitude)]
ndbc_wind_stations  = ndbc_wind_stations   [get_mask(bbox,ndbc_wind_stations.Longitude,ndbc_wind_stations.Latitude)]

coops_wlev_stations.to_csv('coops_wlev_stations_hsofs.csv')
coops_mete_stations.to_csv('coops_mete_stations_hsofs.csv')

ndbc_wave_stations.to_csv('ndbc_wave_stations_hsofs.csv')
ndbc_wind_stations.to_csv('ndbc_wind_stations_hsofs.csv')




"""


import xml.etree.cElementTree as et

base_dir = '/data01/data01/01-projects/07-NOAA-CoastalAct/04-working/03-read_wave_data/'


url   = 'https://tidesandcurrents.noaa.gov/mdapi/v0.6/webapi/'
fname = 'stations.xml'
wget.download(url+fname, out = base_dir)

parsed_xml = et.parse(base_dir+fname )

for node in parsed_xml.getroot():
    print node.text

coops2df
df = coops2df()



"""
