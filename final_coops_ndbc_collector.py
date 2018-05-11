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


@retry(stop_max_attempt_number=5, wait_fixed=3000)
def get_coops(start, end, sos_name, units, bbox,datum='MSL', verbose=True):
    """
    function to read COOPS data
    We need to retry in case of failure b/c the server cannot handle
    the high traffic during hurricane season.
    """
    collector = CoopsSos()
    collector.set_bbox(bbox)
    collector.end_time = end
    collector.start_time = start
    collector.variables = [sos_name]
    ofrs = collector.server.offerings
    title = collector.server.identification.title
    config = dict(
        units=units,
        sos_name=sos_name,
        datum='MSL',            ###Saeed added
    )

    data = collector2table(
        collector=collector,
        config=config,
        col='{} ({})'.format(sos_name, units.format(cf_units.UT_ISO_8859_1))
    )

    # Clean the table.
    table = dict(
        station_name = [s._metadata.get('station_name') for s in data],
        station_code = [s._metadata.get('station_code') for s in data],
        sensor       = [s._metadata.get('sensor')       for s in data],
        lon          = [s._metadata.get('lon')          for s in data],
        lat          = [s._metadata.get('lat')          for s in data],
        depth        = [s._metadata.get('depth', 'NA')  for s in data],
    )

    table = pd.DataFrame(table).set_index('station_name')
    if verbose:
        print('Collector offerings')
        print('{}: {} offerings'.format(title, len(ofrs)))
    return data, table


@retry(stop_max_attempt_number=5, wait_fixed=3000)
def get_ndbc(start, end, bbox , sos_name='waves',datum='MSL', verbose=True):
    """
    function to read NBDC data



    sos_name = waves    
    all_col = (['station_id', 'sensor_id', 'latitude (degree)', 'longitude (degree)',
           'date_time', 'sea_surface_wave_significant_height (m)',
           'sea_surface_wave_peak_period (s)', 'sea_surface_wave_mean_period (s)',
           'sea_surface_swell_wave_significant_height (m)',
           'sea_surface_swell_wave_period (s)',
           'sea_surface_wind_wave_significant_height (m)',
           'sea_surface_wind_wave_period (s)', 'sea_water_temperature (c)',
           'sea_surface_wave_to_direction (degree)',
           'sea_surface_swell_wave_to_direction (degree)',
           'sea_surface_wind_wave_to_direction (degree)',
           'number_of_frequencies (count)', 'center_frequencies (Hz)',
           'bandwidths (Hz)', 'spectral_energy (m**2/Hz)',
           'mean_wave_direction (degree)', 'principal_wave_direction (degree)',
           'polar_coordinate_r1 (1)', 'polar_coordinate_r2 (1)',
           'calculation_method', 'sampling_rate (Hz)', 'name'])
    
    sos_name = winds    

    all_col = (['station_id', 'sensor_id', 'latitude (degree)', 'longitude (degree)',
       'date_time', 'depth (m)', 'wind_from_direction (degree)',
       'wind_speed (m/s)', 'wind_speed_of_gust (m/s)',
       'upward_air_velocity (m/s)', 'name'])

    """
    #add remove from above
    if   sos_name == 'waves':
            col = ['sea_surface_wave_significant_height (m)','sea_surface_wave_peak_period (s)',
                   'sea_surface_wave_mean_period (s)','sea_water_temperature (c)',
                   'sea_surface_wave_to_direction (degree)']
    elif sos_name == 'winds':
            col = ['wind_from_direction (degree)','wind_speed (m/s)',
                   'wind_speed_of_gust (m/s)','upward_air_velocity (m/s)']
    
    
    
    collector = NdbcSos()
    collector.set_bbox(bbox)
    collector.start_time = start

    collector.variables = [sos_name]
    ofrs = collector.server.offerings
    title = collector.server.identification.title
    
    collector.features = None
    collector.end_time = start + datetime.timedelta(1)
    response = collector.raw(responseFormat='text/csv')
    
    
    df = pd.read_csv(BytesIO(response), parse_dates=True)
    g = df.groupby('station_id')
    df = dict()
    for station in g.groups.keys():
        df.update({station: g.get_group(station).iloc[0]})
    df = pd.DataFrame.from_dict(df).T
    
    station_dict = {}
    for offering in collector.server.offerings:
        station_dict.update({offering.name: offering.description})
    
    names = []
    for sta in df.index:
        names.append(station_dict.get(sta, sta))
    
    df['name'] = names
    
    #override short time
    collector.end_time = end

    
    data = []
    for k, row in df.iterrows():
        station_id = row['station_id'].split(':')[-1]
        collector.features = [station_id]
        response = collector.raw(responseFormat='text/csv')
        kw = dict(parse_dates=True, index_col='date_time')
        obs = pd.read_csv(BytesIO(response), **kw).reset_index()
        obs = obs.drop_duplicates(subset='date_time').set_index('date_time')
        series = obs[col]
        series._metadata = dict(
            station=row.get('station_id'),
            station_name=row.get('name'),
            station_code=str(row.get('station_id').split(':')[-1]),
            sensor=row.get('sensor_id'),
            lon=row.get('longitude (degree)'),
            lat=row.get('latitude (degree)'),
            depth=row.get('depth (m)'),
        )
    
        data.append(series)
    
    
    # Clean the table.
    table = dict(
        station_name = [s._metadata.get('station_name') for s in data],
        station_code = [s._metadata.get('station_code') for s in data],
        sensor       = [s._metadata.get('sensor')       for s in data],
        lon          = [s._metadata.get('lon')          for s in data],
        lat          = [s._metadata.get('lat')          for s in data],
        depth        = [s._metadata.get('depth', 'NA')  for s in data],
    )
    
    

    table = pd.DataFrame(table).set_index('station_name')
    if verbose:
        print('Collector offerings')
        print('{}: {} offerings'.format(title, len(ofrs)))
    
    return data, table




base_dir = '/disks/NASARCHIVE/saeed_moghimi/data/coops_obs_pyoos/'



# OOI Endurance Array bounding box# OOI E 
bbox = [-127, 43, -123.75, 48]
#South Florida
bbox = [-87 , 24, -79, 30]  

#hsofs
bbox = [-99.0, 5.0, -52.8, 46.3]

#Sandy
name = 'sandy'
year = '2012'
#bbox = [ -82.0, 23.0 , -67.0, 43.6]
#bbox = [ -79.0 ,32.0 , -69.0, 42.0]
bbox = [ -77.0 ,37.0 , -70.0, 42.0]

                

#sandy 
start_dt =  datetime.datetime(2012,10,22)
end_dt   =  datetime.datetime(2012,11,4)




print('  > Get wind ocean information (ndbc)')
wnd_ocn, wnd_ocn_table = get_ndbc(
    start=start_dt,
    end=end_dt,
    sos_name='winds',
    bbox=bbox,
    )

print('  > Get wave ocean information (ndbc)')
wav_ocn, wav_ocn_table = get_ndbc(
    start=start_dt,
    end=end_dt,
    sos_name='waves',
    bbox=bbox,
    )


print('  > Get water level information  CO-OPS')
ssh, ssh_table = get_coops(
    start=start_dt,
    end=end_dt,
    sos_name='water_surface_height_above_reference_datum',
    units=cf_units.Unit('meters'),
    datum = 'MSL',
    bbox=bbox,
)

print('  > Get wind information CO-OPS')
wnd_obs, wnd_obs_table = get_coops(
    start=start_dt,
    end=end_dt,
    sos_name='wind_speed',
    units=cf_units.Unit('m/s'),
    bbox=bbox,
    )


all_data = dict(wnd_ocn =wnd_ocn   , wnd_ocn_table = wnd_ocn_table,
                wav_ocn = wav_ocn  , wav_ocn_table = wav_ocn_table,
                ssh     =  ssh     , ssh_table     = ssh_table,
                wnd_obs =  wnd_obs , wnd_obs_table = wnd_obs_table)


bbox_txt = str(bbox).replace(' ','_').replace(',','_').replace('[','_').replace(']','_')
scr_dir    = base_dir + '/' + name+year+'/'
os.system('mkdir -p ' + scr_dir)
pickname = scr_dir + name+year+bbox_txt+'.pik2'
f = open(pickname, 'wb')
pickle.dump(all_data,f,protocol=2)


# back up script file
args=sys.argv
scr_name = args[0]
os.system('cp -fr  '+scr_name +'    '+scr_dir)




with open(pick, "rb") as f:
    w = pickle.load(f)

f = open(pick, "rb")
w = pickle.load(f)







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

    return df
































coops_wlev_stations = get_coops_stations_info(type = 'wlev')
coops_mete_stations = get_coops_stations_info(type = 'mete')
ndbc_wave_stations  = get_ndbc_stations_info(type = 'wave')
ndbc_wind_stations  = get_ndbc_stations_info(type = 'wind')










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
