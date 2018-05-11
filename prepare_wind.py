
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


try:
    os.system('rm base_info.pyc')
except:
    pass
if 'base_info' in sys.modules:  
    del(sys.modules["base_info"])
import base_info



def find_nearest1d(xvec,yvec,xp,yp):
    """
    In: xvec, yvec of the grid and xp,yp of the point of interst
    Retun: i,j,proximity of the nearset grid point
    """

    dist = np.sqrt((xvec-xp)**2+(yvec-yp)**2)
    i = np.where(dist==dist.min())
    return i[0],dist.min()

def model_on_data(data_dates, model_dates, model_val):
    print '  >>>>>>>>>>>>>>>   '
    units     = 'seconds since 2012-04-01 00:00:00'
    data_sec  = n4.date2num(data_dates , units)
    model_sec = n4.date2num(model_dates, units)
    return np.interp(data_sec, model_sec, model_val)



# Ike
key  = '10-atm:y-tid:y-wav:y'

#Sandy
key  = '01-atm:y-tid:y-wav:n'


##  wave atm data assimilation
wind_inp = glob.glob(base_info.cases[key]['dir'] +'/inp_atmesh/*')[0]
# wave_inp = glob.glob(base_info.cases[key]['dir'] +'/inp_wav/*')[0]


print (wind_inp)
#
ncw  = n4.Dataset(wind_inp,'r')
ncvw = ncw.variables 
lonw = ncvw['longitude'][:]
latw = ncvw['latitude' ][:]
uwnd = ncvw['uwnd'][:]
vwnd = ncvw['vwnd'][:]
pres = ncvw['P']   [:]
wdates = n4.num2date(ncvw['time'][:],units=ncvw['time'].units)

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
    print (ip)
    [i],prox = find_nearest1d(xvec = lonw,yvec = latw, xp = lons[ip],yp = lats[ip])
    uwnds[:,ip] = model_on_data(data_dates=sdates, model_dates=wdates, model_val=uwnd[:,i]) 
    vwnds[:,ip] = model_on_data(data_dates=sdates, model_dates=wdates, model_val=vwnd[:,i])
    press[:,ip] = model_on_data(data_dates=sdates, model_dates=wdates, model_val=pres[:,i])


uwind1 =  nc0.createVariable(varname = 'uwnd', datatype='f8', dimensions=('time','station',))
vwind1 =  nc0.createVariable(varname = 'vwnd', datatype='f8', dimensions=('time','station',))
press1 =  nc0.createVariable(varname = 'pres', datatype='f8', dimensions=('time','station',))

uwind1[:,:] = uwnds[:,:]
vwind1[:,:] = vwnds[:,:]
press1[:,:] = press[:,:]

nc0.close()
#






