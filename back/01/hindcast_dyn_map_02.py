
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


###############################################################
# Original development from https://github.com/ocefpaf/python_hurricane_gis_map
# # Exploring the NHC GIS Data
# 
# This notebook aims to demonstrate how to create a simple interactive GIS map with the National Hurricane Center predictions [1] and CO-OPS [2] observations along the Hurricane's path.
# 
# 
# 1. http://www.nhc.noaa.gov/gis/
# 2. https://opendap.co-ops.nos.noaa.gov/ioos-dif-sos/
# 
# 
# NHC codes storms are coded with 8 letter names:
# - 2 char for region `al` &rarr; Atlantic
# - 2 char for number `11` is Irma
# - and 4 char for year, `2017`
# 
# Browse http://www.nhc.noaa.gov/gis/archive_wsurge.php?year=2017 to find other hurricanes code.
###############################################################


from bs4 import BeautifulSoup
import pandas as pd
import requests
import numpy as np
import string
#
import os
import sys
#
try:
    from urllib.request import urlopen, urlretrieve
except:
    from urllib import urlopen, urlretrieve
import lxml.html
#

from glob import glob
import geopandas
#
import arrow
#

import numpy as np
from shapely.geometry import LineString
#

import cf_units
from ioos_tools.ioos import collector2table
import pandas as pd
from pyoos.collectors.coops.coops_sos import CoopsSos
from pyoos.collectors.ndbc.ndbc_sos import NdbcSos


from retrying import retry
#
import netCDF4
#
import folium
from folium.plugins import Fullscreen, MarkerCluster
from ioos_tools.ioos import get_coordinates
from branca.element import *



import matplotlib as mpl
mpl.use('Agg')
  
import matplotlib.tri as Tri
import matplotlib.pyplot as plt
from shapely.geometry import mapping, Polygon
import fiona


#year    = '2017'
#name    = 'IRMA'
#inp_dir = '../test_storm_irma/'


#year    = '2012'
#name    = 'SANDY'
#inp_dir = '../test_storm_sandy/'

#year    = '2016'
#name    = 'MATTHEW'
#inp_dir = '../test_storm_matthew/'


year    = '2008'
name    = 'IKE'
inp_dir = '../test_storm_ike/'


#year    = '2003'
#name    = 'ISABEL'
#inp_dir = '../test_storm_isabel/'


plot_cones = True
plot_sat   = False

fort61       = inp_dir + 'fort.61.nc'
felev        = inp_dir + 'maxele.63.nc'
fgrd         = inp_dir + 'depth_hsofs_inp.nc'
fhwm         = inp_dir + 'hwm.csv'

print ('\n\n\n storm: ', name, 'Year: ', year, '\n\n\n') 
#year = '2012'
#name = 'SANDY'



############################
from   matplotlib.colors import LinearSegmentedColormap
cdict = {'red': ((0.  , 1  , 1),
                 (0.05, 1  , 1),
                 (0.11, 0  , 0),
                 (0.66, 1, 1),
                 (0.89, 1, 1),
                 (1   , 0.5, 0.5)),
         'green': ((0., 1, 1),
                   (0.05, 1, 1),
                   (0.11, 0, 0),
                   (0.375, 1, 1),
                   (0.64, 1, 1),
                   (0.91, 0, 0),
                   (1, 0, 0)),
         'blue': ((0., 1, 1),
                  (0.05, 1, 1),
                  (0.11, 1, 1),
                  (0.34, 1, 1),
                  (0.65, 0, 0),
                  (1, 0, 0))}

jetMinWi = LinearSegmentedColormap('my_colormap',cdict,256)


my_cmap = plt.cm.jet

###############################################################
#Functions
def url_lister(url):
    urls = []
    connection = urlopen(url)
    dom = lxml.html.fromstring(connection.read())
    for link in dom.xpath('//a/@href'):
        urls.append(link)
    return urls


def download(url, path, fname):
    sys.stdout.write(fname + '\n')
    if not os.path.isfile(path):
        urlretrieve(
            url,
            filename=path,
            reporthook=progress_hook(sys.stdout)
        )
        sys.stdout.write('\n')
        sys.stdout.flush()


def progress_hook(out):
    """
    Return a progress hook function, suitable for passing to
    urllib.retrieve, that writes to the file object *out*.
    """

    def it(n, bs, ts):
        got = n * bs
        if ts < 0:
            outof = ''
        else:
            # On the last block n*bs can exceed ts, so we clamp it
            # to avoid awkward questions.
            got = min(got, ts)
            outof = '/%d [%d%%]' % (ts, 100 * got // ts)
        out.write("\r  %d%s" % (got, outof))
        out.flush()
    return it



def get_nhc_storm_info (year,name):
    """
    
    """

    print('Read list of hurricanes from NHC based on year')
    
    if int(year) < 2008:  
        print ('  ERROR:   GIS Data is not available for storms before 2008 ')
        sys.exit('Exiting .....')
    
     
    url = 'http://www.nhc.noaa.gov/gis/archive_wsurge.php?year='+year

    r = requests.get(url)

    soup = BeautifulSoup(r.content, 'lxml')

    table = soup.find('table')
    #table = [row.get_text().strip().split(maxsplit=1) for row in table.find_all('tr')]

    tab = []
    for row in table.find_all('tr'):
        tmp = row.get_text().strip().split()
        tab.append([tmp[0],tmp[-1]])
    
    print (tab)   

    df = pd.DataFrame(
        data=tab[:],
        columns=['identifier', 'name'],
    ).set_index('name')


    ###############################

    print('  > based on specific storm go fetch gis files')
    hid = df.to_dict()['identifier'][name]
    code = ('{}'+year).format(hid)
    hurricane_gis_files = '{}_5day'.format(code)
    
    return code,hurricane_gis_files



def download_nhc_gis_files(hurricane_gis_files):
    """
    
    """
    
    nhc = 'http://www.nhc.noaa.gov/gis/forecast/archive/'

    # We don't need the latest file b/c that is redundant to the latest number.
    fnames = [
        fname for fname in url_lister(nhc)
        if fname.startswith(hurricane_gis_files) and 'latest' not in fname
    ]


    base = os.path.abspath(
        os.path.join(os.path.curdir, 'data', hurricane_gis_files)
    )

    if not os.path.exists(base):
        os.makedirs(base)

    for fname in fnames:
        url = '{}/{}'.format(nhc, fname)
        path = os.path.join(base, fname)
        download(url, path,fname)

    return base
    #################################




# Only needed to run on binder!
# See https://gitter.im/binder-project/binder?at=59bc2498c101bc4e3acfc9f1
os.environ['CPL_ZIP_ENCODING'] = 'UTF-8'

def read_advisory_cones_info(hurricane_gis_files,base):
    print('  >  Read cones shape file ...')

    cones, points = [], []
    for fname in sorted(glob(os.path.join(base, '{}_*.zip'.format(hurricane_gis_files)))):
        number = os.path.splitext(os.path.split(fname)[-1])[0].split('_')[-1]
        
        # read cone shapefiles
        
        if int(year) < 2014:
            #al092008.001_5day_pgn.shp
            divd =  '.'
        else:
            divd =  '-'
        
        pgn = geopandas.read_file(
            ('/{}'+divd+'{}_5day_pgn.shp').format(code, number),
            vfs='zip://{}'.format(fname)
        )
        cones.append(pgn)
        
        #read points shapefiles
        pts = geopandas.read_file(
            ('/{}'+divd+'{}_5day_pts.shp').format(code, number),
            vfs='zip://{}'.format(fname)
        )
        # Only the first "obsevartion."
        points.append(pts.iloc[0])
    
    return cones,points,pts


@retry(stop_max_attempt_number=5, wait_fixed=3000)
def get_coops(start, end, sos_name, units, bbox, verbose=False):
    """
    function to read COOPS data
    We need to retry in case of failure b/c the server cannot handle
    the high traffic during hurricane season.
    """
    print('     >> Get CO-OPS information')

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
def get_ndbc(start, end, sos_name, units, bbox, verbose=False):
    """
    function to read NDBC data
    We need to retry in case of failure b/c the server cannot handle
    the high traffic during hurricane season.
    """
    print('     >> Get NDBC information')

collector = NdbcSos()
collector.set_bbox(bbox)
collector.end_time = end
collector.start_time = start
collector.variables = [sos_name]
ofrs = collector.server.offerings
title = collector.server.identification.title

fmt = '{:*^64}'.format

print(fmt(' NDBC Collector offerings '))
print('{}: {} offerings'.format(title, len(ofrs)))



https://ioos.github.io/system-test/blog/2015/10/12/fetching_data/








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



buoy, buoy_table = get_ndbc(
    start=start_dt,
    end=end_dt,
    sos_name='sea_water_temperature',
    units=cf_units.Unit('celsius'),
    bbox=bbox,
)






######################################
# Let's create a color code for the point track.

colors = {
    'subtropical depression': '#ffff99',
    'tropical depression': '#ffff66',
    'tropical storm': '#ffcc99',
    'subtropical storm': '#ffcc66',
    'hurricane': 'red',
    'major hurricane': 'crimson',
}
#######################################
############################################################
# plot ssh to pop up when click on obs locations
##
from bokeh.resources import CDN
from bokeh.plotting import figure
from bokeh.embed import file_html
from bokeh.models import Range1d, LinearAxis, HoverTool

from folium import IFrame

# Plot defaults.
tools = "pan,box_zoom,reset"
width, height = 750, 250


def make_plot_wind(ssh, wind):
    p = figure(toolbar_location='above',
               x_axis_type='datetime',
               width=width,
               height=height,
               tools=tools,
               title=ssh.name)

    p.yaxis.axis_label = 'wind speed (m/s)'

    l0 = p.line(
        x=wind.index,
        y=wind.values,
        line_width=5,
        line_cap='round',
        line_join='round',
        legend='wind speed (m/s)',
        color='#9900cc',
        alpha=0.5,
    )

    p.extra_y_ranges = {}
    p.extra_y_ranges['y2'] = Range1d(
        start=-1,
        end=3.5
    )

    p.add_layout(
        LinearAxis(
            y_range_name='y2',
            axis_label='ssh (m)'),
        'right'
    )

    l1 = p.line(
        x=ssh.index,
        y=ssh.values,
        line_width=5,
        line_cap='round',
        line_join='round',
        legend='ssh (m)',
        color='#0000ff',
        alpha=0.5,
        y_range_name='y2',
    )

    p.legend.location = 'top_left'

    p.add_tools(
        HoverTool(
            tooltips=[
                ('wind speed (m/s)', '@y'),
            ],
            renderers=[l0],
        ),
        HoverTool(
            tooltips=[
                ('ssh (m)', '@y'),
            ],
            renderers=[l1],
        ),
    )
    return p


def make_plot(obs, model,label):
    
    
    p = figure(toolbar_location='above',
               x_axis_type='datetime',
               width=width,
               height=height,
               tools=tools,
               title=obs.name)

    p.yaxis.axis_label = label

    l0 = p.line(
        x=model.index,
        y=model.values.squeeze(),
        line_width=5,
        line_cap='round',
        line_join='round',
        legend='model',
        color='#9900cc',
        alpha=0.7,
    )
  

    l1 = p.line(
        x=obs.index,
        y=obs.values,
        line_width=5,
        line_cap='round',
        line_join='round',
        legend='obs.',
        color='#0000ff',
        alpha=0.7,
    )
    
    minx = max (model.index.min(),obs.index.min())
    maxx = min (model.index.max(),obs.index.max())    
    
    minx = model.index.min()
    maxx = model.index.max()
    
    
    p.x_range = Range1d(
        start = minx,
        end   = maxx
    )
    
    
    p.legend.location = 'top_left'

    p.add_tools(
        HoverTool(
            tooltips=[
                ('model', '@y'),
            ],
            renderers=[l0],
        ),
        HoverTool(
            tooltips=[
                ('obs.', '@y'),
            ],
            renderers=[l1],
        ),
    )
    return p




def make_marker(p, location, fname , color = 'green'):
    html = file_html(p, CDN, fname)
    iframe = IFrame(html, width=width+45, height=height+80)

    popup = folium.Popup(iframe, max_width=2650)
    icon = folium.Icon(color = color, icon='stats')
    marker = folium.Marker(location=location,
                           popup=popup,
                           icon=icon)
    return marker

###############################
###try to add countor to map


def Read_maxele_return_plot_obj(fgrd=fgrd,felev=felev):
    """
    
    """
    
    ncg   = netCDF4.Dataset(fgrd)
    ncgv  = ncg.variables
    #read maxelev
    nc0     = netCDF4.Dataset(felev)
    ncv0    = nc0.variables
    data    = ncv0['zeta_max'][:]
    dep0    = ncv0['depth'][:]
    lon0    = ncv0['x'][:]
    lat0    = ncv0['y'][:]
    elems   = ncgv['element'][:,:]-1  # Move to 0-indexing by subtracting 1
    data[data.mask] = 0.0

    MinVal    = np.min(data) 
    MaxVal    = np.max(data)
    NumLevels = 21

    if True:
        MinVal    = max(MinVal,1)
        MaxVal    = min(MaxVal,4)
        NumLevels = 21


    levels = np.linspace(MinVal, MaxVal, num=NumLevels)
    tri = Tri.Triangulation(lon0,lat0, triangles=elems)

    contour = plt.tricontourf(tri, data,levels=levels,cmap = my_cmap ,extend='max')
    return contour,MinVal,MaxVal,levels
#############################################################
#############################################################
from geopandas import GeoDataFrame

def collec_to_gdf(collec_poly):
    """Transform a `matplotlib.contour.QuadContourSet` to a GeoDataFrame"""
    polygons, colors = [], []
    for i, polygon in enumerate(collec_poly.collections):
        mpoly = []
        for path in polygon.get_paths():
            try:
                path.should_simplify = False
                poly = path.to_polygons()
                # Each polygon should contain an exterior ring + maybe hole(s):
                exterior, holes = [], []
                if len(poly) > 0 and len(poly[0]) > 3:
                    # The first of the list is the exterior ring :
                    exterior = poly[0]
                    # Other(s) are hole(s):
                    if len(poly) > 1:
                        holes = [h for h in poly[1:] if len(h) > 3]
                mpoly.append(Polygon(exterior, holes))
            except:
                print('Warning: Geometry error when making polygon #{}'
                      .format(i))
        if len(mpoly) > 1:
            mpoly = MultiPolygon(mpoly)
            polygons.append(mpoly)
            colors.append(polygon.get_facecolor().tolist()[0])
        elif len(mpoly) == 1:
            polygons.append(mpoly[0])
            colors.append(polygon.get_facecolor().tolist()[0])
    return GeoDataFrame(
        geometry=polygons,
        data={'RGBA': colors},
        crs={'init': 'epsg:4326'})


def convert_to_hex(rgba_color) :
    red = str(hex(int(rgba_color[0]*255)))[2:].capitalize()
    green = str(hex(int(rgba_color[1]*255)))[2:].capitalize()
    blue = str(hex(int(rgba_color[2]*255)))[2:].capitalize()

    if blue=='0':
        blue = '00'
    if red=='0':
        red = '00'
    if green=='0':
        green='00'

    return '#'+ red + green + blue



########################################
####       MAIN CODE from HERE     #####
########################################

#read file info
code,hurricane_gis_files = get_nhc_storm_info (year,name)

#donload gis zip files
base = download_nhc_gis_files(hurricane_gis_files)

# get advisory cones and track points
cones,points,pts = read_advisory_cones_info(hurricane_gis_files,base)


#####################################
# Now we can get all the information we need from those GIS files. Let's start with the event dates.
######################################

# We are ignoring the timezone, like AST (Atlantic Time Standar) b/c
# those are not a unique identifiers and we cannot disambiguate.

if  'FLDATELBL' in points[0].keys():
    start = points[0]['FLDATELBL']
    end = points[-1]['FLDATELBL']
    date_key = 'FLDATELBL'

    #start = arrow.get(start, 'YYYY-MM-DD h:mm A ddd').naive
    #end   = arrow.get(end, 'YYYY-MM-DD h:mm A ddd').naive

    start_dt = arrow.get(start, 'YYYY-MM-DD h:mm A ddd').datetime
    end_dt   = arrow.get(end,   'YYYY-MM-DD h:mm A ddd').datetime
elif 'ADVDATE' in points[0].keys():
    #older versions (e.g. IKE)
    start = points[0]['ADVDATE']
    end = points[-1]['ADVDATE']
    date_key = 'ADVDATE'
    
    start_dt = arrow.get(start, 'YYMMDD/hhmm').datetime
    end_dt   = arrow.get(end,   'YYMMDD/hhmm').datetime
else:
    print ('Check for correct time stamp and adapt the code !')
    sys.exit('ERROR') 
#####################################
# Find the bounding box to search the data.

last_cone = cones[-1]['geometry'].iloc[0]
track = LineString([point['geometry'] for point in points])

bounds = np.array([last_cone.buffer(2).bounds, track.buffer(2).bounds]).reshape(4, 2)
lons, lats = bounds[:, 0], bounds[:, 1]
bbox = lons.min(), lats.min(), lons.max(), lats.max()

# Note that the bounding box is derived from the track and the latest prediction cone.
strbbox = ', '.join(format(v, '.2f') for v in bbox)
print('bbox: {}\nstart: {}\n  end: {}'.format(strbbox, start, end))

############################################################

#Read COOPS data
print('  > Get water level information')
ssh, ssh_table = get_coops(
    start=start_dt,
    end=end_dt,
    sos_name='water_surface_height_above_reference_datum',
    units=cf_units.Unit('meters'),
    bbox=bbox,
)

ssh_table 
############################################################

def get_station_ssh(fort61):
    """
        Read model ssh
    """
    nc0      = netCDF4.Dataset(fort61)
    ncv0     = nc0.variables 
    sta_lon  = ncv0['x'][:]
    sta_lat  = ncv0['y'][:]
    sta_nam  = ncv0['station_name'][:].squeeze()
    sta_zeta = ncv0['zeta']        [:].squeeze()
    sta_date = netCDF4.num2date(ncv0['time'][:], ncv0['time'].units)

    stationIDs = []
    mod    = []
    ind = np.arange(len(sta_lat))
    for ista in ind:
        stationID = sta_nam[ista].tostring().decode().rstrip()
        stationIDs.append(stationID)
        mod_tmp = pd.DataFrame(data = np.c_[sta_date, sta_zeta[:,ista]], columns=['date_time',  'ssh']).set_index('date_time')
        mod_tmp._metadata = stationID
        mod.append(mod_tmp)

    stationIDs = np.array(stationIDs)
    mod_table = pd.DataFrame(data = np.c_[ind, stationIDs], columns=['ind',  'station_code'])
   
    return mod,mod_table
    


def get_station_wnd_all(fort61):
    """
    Read model wind
    
    """
    nc0      = netCDF4.Dataset(fort61)
    ncv0     = nc0.variables 
    sta_lon  = ncv0['x'][:]
    sta_lat  = ncv0['y'][:]
    sta_nam  = ncv0['station_name'][:].squeeze()
    sta_uwnd = ncv0['uwnd']        [:].squeeze()
    sta_vwnd = ncv0['vwnd']        [:].squeeze()
    sta_pres = ncv0['pres']        [:].squeeze()
    sta_date = netCDF4.num2date(ncv0['time'][:], ncv0['time'].units)

    stationIDs = []
    mod    = []
    ind = np.arange(len(sta_lat))
    for ista in ind:
        stationID = sta_nam[ista].tostring().decode().rstrip()
        stationIDs.append(stationID)
        mod_tmp = pd.DataFrame(data = np.c_[sta_date,sta_uwnd[:,ista],sta_vwnd[:,ista],sta_pres[:,ista]],
                               columns = ['date_time', 'uwnd' , 'vwnd', 'pres']).set_index('date_time')
        mod_tmp._metadata = stationID
        mod.append(mod_tmp)

    stationIDs = np.array(stationIDs)
    mod_table = pd.DataFrame(data = np.c_[ind, stationIDs], columns=['ind',  'station_code'])
   
    return mod,mod_table
    

def get_station_wnd(fort61):
    """
    Read model wind
    
    """
    nc0      = netCDF4.Dataset(fort61)
    ncv0     = nc0.variables 
    sta_lon  = ncv0['x'][:]
    sta_lat  = ncv0['y'][:]
    sta_nam  = ncv0['station_name'][:].squeeze()
    sta_wnd =  np.sqrt ( ncv0['uwnd'] [:].squeeze() ** 2 +  ncv0['vwnd']        [:].squeeze() ** 2 )
    sta_date = netCDF4.num2date(ncv0['time'][:], ncv0['time'].units)

    stationIDs = []
    mod    = []
    ind = np.arange(len(sta_lat))
    for ista in ind:
        stationID = sta_nam[ista].tostring().decode().rstrip()
        stationIDs.append(stationID)
        mod_tmp = pd.DataFrame(data = np.c_[sta_date,sta_wnd[:,ista]],
                               columns = ['date_time', 'wnd' ]).set_index('date_time')
        mod_tmp._metadata = stationID
        mod.append(mod_tmp)

    stationIDs = np.array(stationIDs)
    mod_table = pd.DataFrame(data = np.c_[ind, stationIDs], columns=['ind',  'station_code'])
   
    return mod,mod_table
    

############ Read SSH data
mod,mod_table = get_station_ssh(fort61)

############# Sea Surface height analysis ########################
# For simplicity we will use only the stations that have both wind speed and sea surface height and reject those that have only one or the other.
common  = set(ssh_table['station_code']).intersection(mod_table  ['station_code'].values)

ssh_obs, mod_obs = [], []
for station in common:
    ssh_obs.extend([obs for obs in ssh   if obs._metadata['station_code'] == station])
    mod_obs.extend([obm for obm in mod   if obm._metadata                 == station])


index = pd.date_range(
    start = start_dt.replace(tzinfo=None),
    end   = end_dt.replace  (tzinfo=None),
    freq='15min'
)
#############################################################
#organize and re-index both observations
# Re-index and rename series.
ssh_observations = []
for series in ssh_obs:
    _metadata = series._metadata
    obs = series.reindex(index=index, limit=1, method='nearest')
    obs._metadata = _metadata
    obs.name = _metadata['station_name']
    ssh_observations.append(obs)

##############################################################
#model
model_observations = []
for series in mod_obs:
    _metadata = series._metadata
    obs = series.reindex(index=index, limit=1, method='nearest')
    obs._metadata = _metadata
    obs.name = _metadata
    obs['ssh'][np.abs(obs['ssh']) > 10] = np.nan
    obs.dropna(inplace=True)
    model_observations.append(obs)

############# Wind obs and model analysis ########################
try:
    #read wind model data
    wnd_mod,wnd_mod_table = get_station_wnd(fort61)

    print('  > Get wind information')
    wnd_obs, wnd_obs_table = get_coops(
        start=start_dt,
        end=end_dt,
        sos_name='wind_speed',
        units=cf_units.Unit('m/s'),
        bbox=bbox,
    )

    wnd_obs_table

    # For simplicity we will use only the stations that have both wind speed and sea surface height and reject those that have only one or the other.
    commonw  = set(wnd_obs_table['station_code']).intersection(wnd_mod_table  ['station_code'].values)

    wobs, wmod = [], []
    for station in commonw:
        wobs.extend([obs for obs in wnd_obs   if obs._metadata['station_code'] == station])
        wmod.extend([obm for obm in wnd_mod   if obm._metadata                 == station])


    index = pd.date_range(
        start = start_dt.replace(tzinfo=None),
        end   = end_dt.replace  (tzinfo=None),
        freq='15min'
    )
    #############################################################
    #organize and re-index both observations
    # Re-index and rename series.
    wnd_observs = []
    for series in wobs:
        _metadata = series._metadata
        obs = series.reindex(index=index, limit=1, method='nearest')
        obs._metadata = _metadata
        obs.name = _metadata['station_name']
        wnd_observs.append(obs)

    ##############################################################
    #model
    wnd_models = []
    for series in wmod:
        _metadata = series._metadata
        obs = series.reindex(index=index, limit=1, method='nearest')
        obs._metadata = _metadata
        obs.name = _metadata
        #obs['ssh'][np.abs(obs['ssh']) > 10] = np.nan
        obs.dropna(inplace=True)
        wnd_models.append(obs)
    wind_stations = True  
except:
    print (' >  fort.61 does not include wind info ..')
    wind_stations = False
#######################################################
print('  > Put together the final maps')
# Here is the final result. Explore the map by clicking on the map features plotted!
lon = track.centroid.x
lat = track.centroid.y
############################################################


#####################################################################
#if  'FLDATELBL' in points[0].keys():
##
m = folium.Map(location=[lat, lon], tiles='OpenStreetMap', zoom_start=4)
Fullscreen(position='topright', force_separate_button=True).add_to(m)

print('     > Maxele plot..')
contour,MinVal,MaxVal,levels = Read_maxele_return_plot_obj(fgrd=fgrd,felev=felev)
gdf = collec_to_gdf(contour) # From link above
plt.close('all')

## Get colors in Hex
colors_elev = []
for i in range (len(gdf)):
    color = my_cmap(i / len(gdf)) 
    colors_elev.append( mpl.colors.to_hex(color)   )

#assign to geopandas obj
gdf['RGBA'] = colors_elev
#
#plot geopandas obj
maxele = folium.GeoJson(
    gdf,
    name='Maximum water level from MSL [m]',
    style_function=lambda feature: {
        'fillColor': feature['properties']['RGBA'],
        'color' : feature['properties']['RGBA'],
        'weight' : 1.0,
        'fillOpacity' : 0.6,
        'line_opacity' : 0.6,
        }
    )
    
maxele.add_to(m)
    
#Add colorbar     
color_scale = folium.StepColormap(colors_elev,
    #index=color_domain,
    vmin=MinVal,
    vmax=MaxVal,
    caption= 'Maximum water level from MSL [m]',
    )
m.add_child(color_scale)
#
####################
print('     > CO-Ops stations plot')
# ssh Observations stations 
marker_cluster_coops = MarkerCluster(name='CO-OPs observations')
marker_cluster_coops.add_to(m)
for ssh1, model1 in zip(ssh_observations, model_observations):
    fname = ssh1._metadata['station_code']
    location = ssh1._metadata['lat'], ssh1._metadata['lon']
    p = make_plot(ssh1, model1, 'SSH [m]')
    #p = make_plot(ssh1, ssh1)    
    marker = make_marker(p, location=location, fname=fname)
    marker.add_to(marker_cluster_coops)

####################
if wind_stations:
    # Wind Observations stations.
    #marker_clusterw = MarkerCluster(name='Wind observations')
    #marker_clusterw.add_to(m)
    for ssh1, model1 in zip(wnd_observs,wnd_models):
        fname = ssh1._metadata['station_code']
        location = ssh1._metadata['lat'] , ssh1._metadata['lon'] 
        p = make_plot(ssh1, model1,'Wind [m/s]')
        #p = make_plot(ssh1, ssh1)    
        marker = make_marker(p, location=location, fname=fname, color = 'red')
        marker.add_to(marker_cluster_coops)

###################        
# folium.LayerControl().add_to(m)
p = folium.PolyLine(get_coordinates(bbox),
                    color='#009933',
                    weight=1,
                    opacity=0.5)

p.add_to(m)
#################################### 
# 

###################
print('     > Plot NHC cone predictions')

if plot_cones:
    marker_cluster1 = MarkerCluster(name='Past predictions')
    marker_cluster1.add_to(m)
    def style_function(feature):
        return {
            'fillOpacity': 0,
            'color': 'black',
            'stroke': 1,
            'weight': 0.5,
            'opacity': 0.1,
        }

    track_radius = 4
    
    # Latest cone prediction.
    latest = cones[-1]
    ###
    if  'FLDATELBL' in points[0].keys():    #Newer storms have this information
        names = 'Cone prediction as of {}'.format(latest['ADVDATE'].values[0]),
    else:
        names = 'Cone prediction'
    ###
    folium.GeoJson(
        data=latest.__geo_interface__,
        name=names
    ).add_to(m)
    ###
    # Past cone predictions.
    for cone in cones[:-1]:
        folium.GeoJson(
            data=cone.__geo_interface__,
            style_function=style_function,
        ).add_to(marker_cluster1)

    # Latest points prediction.
    for k, row in pts.iterrows():
        
        if  'FLDATELBL' in points[0].keys():    #Newer storms have this information
            date = row[date_key] 
            hclass = row['TCDVLP']
            popup = '{}<br>{}'.format(date, hclass)
            color = colors[hclass.lower()]
        else:
            popup = '{}<br>{}'.format(name,year )
            color = colors['hurricane']
      
        location = row['LAT'], row['LON']
        folium.CircleMarker(
            location=location,
            radius=track_radius,
            fill=True,
            color=color,
            popup=popup,
        ).add_to(m)

print('     > Plot points along the final track')
#marker_cluster3 = MarkerCluster(name='Track')
#marker_cluster3.add_to(m)

for point in points:
    if  'FLDATELBL' in points[0].keys():    #Newer storms have this information
        date = point[date_key]
        hclass = point['TCDVLP']
        popup = '{}<br>{}'.format(date, hclass)
        color = colors[hclass.lower()]
    else:
        popup = '{}<br>{}'.format(name,year )
        color = colors['hurricane']
    
    location = point['LAT'], point['LON']
    folium.CircleMarker(
        location=location,
        radius=track_radius,
        fill=True,
        color=color,
        popup=popup,
    ).add_to(m)


print('     > Plot High Water Marks')
df = pd.read_csv(fhwm)
lon_hwm = df.lon.values
lat_hwm = df.lat.values
hwm     = df.elev_msl_m.values
#
#marker_cluster_hwm = MarkerCluster(name='High Water Marks')
#marker_cluster_hwm.add_to(m)
# HWMs
for im in range (len(hwm)):
#for im in np.arange(100):
    location = lat_hwm[im], lon_hwm[im]
    ind = np.argmin (np.abs(levels-hwm[im]))

    popup = 'HWM_ID:{}<br>{}'.format(df['HWM_ID'][im],df['Description'][im])
    #popup = popup.replace('"',' ').replace('/', ' ').replace(' ','-') 
    popup = popup[:50]
    #print (popup)
    
    folium.CircleMarker(
        location=location,
        radius=3,
        fill=True,
        fill_color = colors_elev[ind],
        fill_opacity = 85,
        color      = colors_elev[ind],
        popup = popup,
    ).add_to(m)

folium.LayerControl().add_to(m)


#################################################
print ('     > Add disclaimer and storm name ...')

Disclaimer_html =   '''
                <div style="position: fixed; 
                            bottom: 15px; left: 20px; width: 520px; height: 40px; 
                            border:2px solid grey; z-index:9999; font-size:12px;
                            ">&nbsp; For Official Use Only. Pre-Decisional information not releasable outside US Government. <br>
                              &nbsp; Contact: CSDL/OCS/NOS/NOAA &nbsp; <br>
                </div>
                ''' 

m.get_root().html.add_child(folium.Element(Disclaimer_html))



storm_info_html ='''
                <div style="position: fixed; 
                            bottom: 75px; left: 20px; width: 150px; height: 50px; 
                            border:2px solid black; z-index:9999; font-size:18px;
                            ">&nbsp; Storm: {} <br>
                              &nbsp; Year:  {}  &nbsp; <br>
                </div>
                '''.format(name,year) 

m.get_root().html.add_child(folium.Element(storm_info_html))
###################################################




fname = '{}_storm.html'.format(name.split()[-1].lower())
m.save(fname)






###test
#####################################################################
#if  'FLDATELBL' in points[0].keys():
##


m = folium.Map(location=[lat, lon], tiles='OpenStreetMap', zoom_start=4)
Fullscreen(position='topright', force_separate_button=True).add_to(m)


if False:
    print('     > Plot High Water Marks')
    df = pd.read_csv(fhwm)
    lon_hwm = df.lon.values
    lat_hwm = df.lat.values
    hwm     = df.elev_msl_m.values
    #
    #marker_cluster_hwm = MarkerCluster(name='High Water Marks')
    #marker_cluster_hwm.add_to(m)
    # HWMs
    for im in range (len(hwm)):
    #for im in np.arange(100):
        location = lat_hwm[im], lon_hwm[im]
        ind = np.argmin (np.abs(levels-hwm[im]))

        popup = 'HWM_ID:{} <br> {}'.format(df['HWM_ID'][im],df['Description'][im])
        #popup = popup.replace('"',' ').replace('/', ' ').replace(' ', '-')
        #print (popup)
        popup = popup[:50]
        #print (popup)
        
        folium.CircleMarker(
            location=location,
            radius=3,
            fill=True,
            fill_color = colors_elev[ind],
            fill_opacity = 85,
            color      = colors_elev[ind],
            popup = popup,
        ).add_to(m)


Disclaimer_html =   '''
                <div style="position: fixed; 
                            bottom: 15px; left: 20px; width: 520px; height: 40px; 
                            border:2px solid grey; z-index:9999; font-size:12px;
                            ">&nbsp; For Official Use Only. Pre-Decisional information not releasable outside US Government. <br>
                              &nbsp; Contact: CSDL/OCS/NOS/NOAA &nbsp; <br>
                </div>
                ''' 

m.get_root().html.add_child(folium.Element(Disclaimer_html))



storm_info_html ='''
                <div style="position: fixed; 
                            bottom: 75px; left: 20px; width: 150px; height: 50px; 
                            border:2px solid black; z-index:9999; font-size:18px;
                            ">&nbsp; Storm: {} <br>
                              &nbsp; Year:  {}  &nbsp; <br>
                </div>
                '''.format(name,year) 

m.get_root().html.add_child(folium.Element(storm_info_html))



folium.LayerControl().add_to(m)

fname = '{}_hwm_storm22.html'.format(name.split()[-1].lower())
m.save(fname)







""""
html="" "
    <h1> This is a big popup</h1><br>
    With a few lines of code...
    <p>
    <code>
        from numpy import *<br>
        exp(-2*pi)
    </code>
    </p>
    "" "

iframe = folium.IFrame(html=html, width=500, height=300)
popup = folium.Popup(iframe, max_width=2650)

folium.Marker([30,-100], popup=popup).add_to(m)

#https://github.com/bobhaffner/medium_posts/blob/master/folium_legend/folium_legend.ipynb
"""
