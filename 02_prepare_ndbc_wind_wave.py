from __future__ import print_function, division
# coding: utf-8


"""
Dynamic map hindcast implementation 
"""
__author__ = "Saeed Moghimi"
__copyright__ = "Copyright 2018, UCAR/NOAA"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "moghimis@gmail.com"

#Thu 19 Apr 2018 03:08:06 PM EDT 



import netCDF4 as n4
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import datetime
import glob
import time
import cPickle as pickle
import pandas as pd


### simulation information
try:
    os.system('rm base_info.pyc')
except:
    pass
if 'base_info' in sys.modules:  
    del(sys.modules["base_info"])
import base_info


#html folium and observation information
try:
    os.system('rm __pycache__/base_info_folium*.pyc'  )
    os.system('rm base_info_folium*.pyc'  )
except:
    pass
if 'base_info_folium' in sys.modules:  
    del(sys.modules["base_info_folium"])
from base_info_folium import *


def find_nearest1d(xvec,yvec,xp,yp):
    """
    In: xvec, yvec of the grid and xp,yp of the point of interst
    Retun: i,j,proximity of the nearset grid point
    """
    dist = np.sqrt((xvec-xp)**2+(yvec-yp)**2)
    i = np.where(dist==dist.min())
    return i[0],dist.min()

def model_on_data(data_dates, model_dates, model_val):
    #print ('  >>>>>>>>>>>>>>>   ')
    units     = 'seconds since 2012-04-01 00:00:00'
    data_sec  = n4.date2num(data_dates , units)
    model_sec = n4.date2num(model_dates, units)
    return np.interp(data_sec, model_sec, model_val)

# Ike
#key  = '10-atm:y-tid:y-wav:y'
#Sandy
#key  = '01-atm:y-tid:y-wav:n'
#key  = '02-atm:y-tid:y-wav:y-try01'
#########################################################
#name  = 'SANDY'
#year  = '2012'

#name  = 'IRENE'
#year  = '2011'

print (' > Read Obs stas ...')
sta_info_dir = '/scratch4/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/coops_ndbc_data/' 
ndbc_wave_sta_file = os.path.join(sta_info_dir,name+year,'ndbc_wave','table.csv')
ndbc_wind_sta_file = os.path.join(sta_info_dir,name+year,'ndbc_wind','table.csv')

wave_sta = pd.read_csv(ndbc_wave_sta_file)
wind_sta = pd.read_csv(ndbc_wind_sta_file)
#make numeric > numeric
for df in [wind_sta,wave_sta]:
    for col in df.columns:
         try:
             df[col]  = df[col].astype('float64') 
         except:
             pass
nloc_wind = len (wind_sta)
nloc_wave = len (wave_sta)
######################################################

for key in base_info.cases.keys():
    print ('>>> ',key ,  base_info.cases[key]['label'])  
    ########
    wind_inp = glob.glob(base_info.cases[key]['dir'] +'/inp_atmesh/*')
    wind_out = base_info.cases[key]['dir']+'/fort.74.nc'
    
    print (' > Wind results ...')
    if len(wind_inp) > 0 or os.path.exists(wind_out):
        if len(wind_inp) > 0:
            ncw  = n4.Dataset(wind_inp,'r')
            ncvw = ncw.variables 
            lonw = ncvw['longitude'][:]
            latw = ncvw['latitude' ][:]
            uwnd = ncvw['uwnd'][:]
            vwnd = ncvw['vwnd'][:]
            pres = ncvw['P']   [:]
            wind_dates = n4.num2date(ncvw['time'][:],units=ncvw['time'].units)
            nt_wind = len(wind_dates)

        elif os.path.exists(wind_out):
            #
            ncw  = n4.Dataset(wind_out,'r')
            ncvw = ncw.variables 
            lonw = ncvw['x'][:]
            latw = ncvw['y' ][:]
            uwnd = ncvw['windx'][:]
            vwnd = ncvw['windy'][:]
            wind_dates = n4.num2date(ncvw['time'][:],units=ncvw['time'].units)
            nt_wind = len(wind_dates)  
            #
            ncp  = n4.Dataset(base_info.cases[key]['dir']+'/fort.73.nc','r')                 
            pres = ncp.variables['pressure'][:]
            ncp.close()

        #
        uwnds = np.zeros ((nt_wind,nloc_wind))
        vwnds = np.zeros_like(uwnds)
        press = np.zeros_like(uwnds)  
        #
        lons    = wind_sta.lon
        lats    = wind_sta.lat
        sat_lab = wind_sta['station_code'][:]
        for ip in range(len(lons)):
            [i],prox = find_nearest1d(xvec = lonw,yvec = latw, xp = lons[ip],yp = lats[ip])
            uwnds[:,ip] = uwnd[:,i]   
            vwnds[:,ip] = vwnd[:,i]  
            press[:,ip] = pres[:,i]  
        
        wind_out_fname = base_info.cases[key]['dir'] + '/01_wind_on_ndbc_obs.nc'
        print (wind_out_fname)
        
        nc = n4.Dataset(wind_out_fname,'w')
         
        nc.Description = 'Wind model for obs locations'
        nc.Author = 'moghimis@gmail.com'
        nc.Created = datetime.datetime.now().isoformat()
         
        # DIMENSIONS #
       
        nc.createDimension('time', None)
        nc.createDimension('station'   , nloc_wind)
        nc.createDimension('namelen'   ,  50 )

        name1       = nc.createVariable(varname = 'station_name', datatype='S1', dimensions=('station','namelen',))
        for ista in range(len(sat_lab)):
            label =  sat_lab[ista]
            #print (label)
            for ich in range(len(label)):
                name1[ista,ich]    = label[ich]
        
        time1       = nc.createVariable(varname = 'time', datatype='f8', dimensions=('time',))
        time1.units = ncvw['time'].units
        time1[:]    = ncvw['time'][:]  
         
        lon1       = nc.createVariable(varname = 'lon', datatype='f8', dimensions=('station',))
        lon1[:]    =   lons.values [:]

        lat1       = nc.createVariable(varname = 'lat', datatype='f8', dimensions=('station',))
        lat1[:]    = lats.values [:]
                 
        uwind1 =  nc.createVariable(varname = 'uwnd', datatype='f8', dimensions=('time','station',))
        vwind1 =  nc.createVariable(varname = 'vwnd', datatype='f8', dimensions=('time','station',))
        press1 =  nc.createVariable(varname = 'pres', datatype='f8', dimensions=('time','station',))
        
        uwind1[:,:] = uwnds[:,:]
        vwind1[:,:] = vwnds[:,:]
        press1[:,:] = press[:,:]
        
        nc.close()
        ncw.close()
        #sys.exit()
    
        #########################################################
        print (' > Read CO-OPS Obs stations ...')
        ## We add wind from model at the same place as water elevation
        #
        tmp    = base_info.cases[key]['dir'] + '/fort.61.nc'
        fort61 = base_info.cases[key]['dir'] + '/fort_wind.61.nc'
        #
        os.system('cp -rf ' + tmp + '  ' + fort61)
        nc0 = n4.Dataset(fort61,'r+')
        ncv0 = nc0.variables 
        lons = ncv0['x'][:]
        lats = ncv0['y'][:]
        zeta = ncv0['zeta'][:]
        sdates = n4.num2date(ncv0['time'][:],units=ncv0['time'].units)
        
        #
        uwnds = np.zeros_like(zeta)
        vwnds = np.zeros_like(zeta)
        press = np.zeros_like(zeta)  
        #
        for ip in range(len(lons)):
            #print (ip)
            [i],prox = find_nearest1d(xvec = lonw,yvec = latw, xp = lons[ip],yp = lats[ip])
            uwnds[:,ip] = model_on_data(data_dates=sdates, model_dates=wind_dates, model_val=uwnd[:,i]) 
            vwnds[:,ip] = model_on_data(data_dates=sdates, model_dates=wind_dates, model_val=vwnd[:,i])
            press[:,ip] = model_on_data(data_dates=sdates, model_dates=wind_dates, model_val=pres[:,i])
        
        
        uwind1 =  nc0.createVariable(varname = 'uwnd', datatype='f8', dimensions=('time','station',))
        vwind1 =  nc0.createVariable(varname = 'vwnd', datatype='f8', dimensions=('time','station',))
        press1 =  nc0.createVariable(varname = 'pres', datatype='f8', dimensions=('time','station',))
        
        uwind1[:,:] = uwnds[:,:]
        vwind1[:,:] = vwnds[:,:]
        press1[:,:] = press[:,:]
        
        nc0.close()    
    
    
    wave_inp = glob.glob(base_info.cases[key]['dir'] +'/inp_wavdata/*')
    if len(wave_inp) > 0:
        print (' > Wave results ...')
        hsig_inp = base_info.cases[key]['hsig_file']
        wdir_inp = base_info.cases[key]['wdir_file']
        ###
        nch  = n4.Dataset(hsig_inp,'r')
        ncvh = nch.variables 
        lonh = ncvh['longitude'][:]
        lath = ncvh['latitude' ][:]
        hsig = ncvh['hs'][:]
        
        wave_dates = n4.num2date(ncvh['time'][:],units=ncvh['time'].units)

        #
        ncd  = n4.Dataset(wdir_inp,'r')
        ncvd = ncd.variables 
        wdir = ncvd['dir'][:]
        ncd.close()
        #
        
        lons = wave_sta.lon
        lats = wave_sta.lat
        sat_lab = wave_sta['station_code'][:]
        hsigs = np.zeros ((len(wave_dates),nloc_wave))
        wdirs = np.zeros_like(hsigs)
        
        for ip in range(len(lons)):
            [i],prox = find_nearest1d(xvec = lonh,yvec = lath, xp = lons[ip],yp = lats[ip])
            hsigs[:,ip] = hsig[:,i]   
            wdirs[:,ip] = wdir[:,i]  
        
        wind_out_fname = base_info.cases[key]['dir'] + '/01_wave_on_ndbc_obs.nc'
        nc = n4.Dataset(wind_out_fname,'w')
         
        nc.Description = 'Wave model for obs locations'
        nc.Author = 'moghimis@gmail.com'
        nc.Created = datetime.datetime.now().isoformat()
         
        # DIMENSIONS #
        nc.createDimension('time', None)
        nc.createDimension('station'   , nloc_wave)
        nc.createDimension('namelen'   ,  50 )

        name1       = nc.createVariable(varname = 'station_name', datatype='S1', dimensions=('station','namelen',))
        for ista in range(len(sat_lab)):
            label =  sat_lab[ista]
            #print (label)
            for ich in range(len(label)):
                name1[ista,ich]    = label[ich]

        
        lon1       = nc.createVariable(varname = 'lon', datatype='f8', dimensions=('station',))
        lon1[:]    =   lons.values [:]

        lat1       = nc.createVariable(varname = 'lat', datatype='f8', dimensions=('station',))
        lat1[:]    = lats.values [:]

        time1       = nc.createVariable(varname = 'time', datatype='f8', dimensions=('time',))
        time1.units = ncvh['time'].units
        time1[:]    = ncvh['time'][:]  
  
        uwind1 =  nc.createVariable(varname = 'hsig', datatype='f8', dimensions=('time','station',))
        vwind1 =  nc.createVariable(varname = 'wdir', datatype='f8', dimensions=('time','station',))
        
        uwind1[:,:] = hsigs[:,:]
        vwind1[:,:] = wdirs[:,:]
        
        nc.close()
        nch.close()

    
    print ('>>> Prepare max surge ..')
    
    fname  =  base_info.cases[base_info.key0]['dir'] + '/maxele.63.nc'
    nc0    = n4.Dataset(fname)
    ncv0   = nc0.variables
    zeta0  = ncv0['zeta_max'][:]
    nc0.close()
    fname_orig  =  base_info.cases[key]['dir'] + '/maxele.63.nc'
    fname       =  base_info.cases[key]['dir'] + '/maxele.63_all.nc'
    os.system('cp -rf ' + fname_orig + '  ' + fname )
    
    nc1     = n4.Dataset(fname,'r+')
    ncv1    = nc1.variables
    zeta1  = ncv1['zeta_max'][:]
    mask = zeta1 < 0
    val = zeta1 - zeta0
    val[np.isnan(val)] = 0.0
    
    node  = nc1.dimensions['node']
    surge1 = nc1.createVariable(varname = 'surge', datatype='f8', dimensions=('node',))
    surge1[:] = val
    surge1.long_name = 'Max surge from max elev minus max tide in meter. Water Above Ground Level.'
    surge1.short_name = 'Surge [m]'
    nc1.close()

print ('Organize and copy files ...')
out_dir = base_info.base_dir_sm + '/z01_4web_plot/' + base_info.storm_name+'/'


for key in np.sort(base_info.cases.keys()):
    subdir = key +'.'+ base_info.cases[key]['label'] +'.'+ base_info.cases[key]['dir'].split('/')[-2] + '/'
    print (subdir)
    out_dir_case = out_dir + subdir
    os.system('mkdir -p ' +  out_dir_case)
    
    
    fnames = [
        base_info.cases[key]['dir']+ '/maxele.63_all.nc',
        base_info.cases[key]['dir']+ '/01_wave_on_ndbc_obs.nc',
        base_info.cases[key]['dir']+ '/01_wind_on_ndbc_obs.nc',
        base_info.cases[key]['dir']+ '/fort_wind.61.nc',
        ]
    
    for fname in fnames:
        os.system('cp -fv ' + fname + ' ' +  out_dir_case  )

# back up script file
args=sys.argv
scr_name = args[0]
os.system('cp -fr  '+scr_name +'     '+out_dir_case)

print ('Finish ...')



