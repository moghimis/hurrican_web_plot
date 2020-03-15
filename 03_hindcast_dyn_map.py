
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


import pandas as pd
import numpy as np
import string
#
import os
import sys
from glob import glob
#
import arrow
#
from shapely.geometry import LineString
import netCDF4
#
import folium
from folium.plugins import Fullscreen, MarkerCluster,MousePosition,FloatImage
from ioos_tools.ioos import get_coordinates
from branca.element import *

import matplotlib as mpl
mpl.use('Agg')
  
import matplotlib.tri as Tri
import matplotlib.pyplot as plt
from shapely.geometry import mapping, Polygon
import fiona

from bokeh.resources import CDN
from bokeh.plotting import figure
from bokeh.models import Title
from bokeh.embed import file_html
from bokeh.models import Range1d, LinearAxis, HoverTool
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource


from folium import IFrame
from geopandas import GeoDataFrame


import pickle

try:
    os.system('rm __pycache__/hurricane_funcs*.pyc'  )
    os.system('rm hurricane_funcs*.pyc'  )
except:
    pass
if 'hurricane_funcs' in sys.modules:  
    del(sys.modules["hurricane_funcs"])
from hurricane_funcs import *


try:
    os.system('rm __pycache__/base_info_folium*.pyc'  )
    os.system('rm base_info_folium*.pyc'  )
except:
    pass
if 'base_info_folium' in sys.modules:  
    del(sys.modules["base_info_folium"])
from base_info_folium import *


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

######################################
# Let's create a color code for the point track.
colors_hurricane_condition = {
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

tools = "pan,box_zoom,reset"
width, height = 750, 250


def make_plot_2axes(ssh, wind):
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



def make_plot_obs(obs,label=None):    
    #TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
    TOOLS="crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,reset,save,"
    
    p = figure(toolbar_location='above',
               x_axis_type='datetime',
               width=width,
               height=height,
               tools=TOOLS)

    p.add_layout(Title(text='Station: '+obs._metadata['station_code'], text_font_style="italic"), 'above')
    p.add_layout(Title(text=obs._metadata['station_name'], text_font_size="10pt"), 'above')

    p.yaxis.axis_label = label

    obs_val = obs.values.squeeze()
    
    l1 = p.line(
        x=obs.index,
        y=obs_val,
        line_width=5,
        line_cap='round',
        line_join='round',
        legend='obs.',
        color='#0000ff',
        alpha=0.7,
    )
    

    minx = obs.index.min()
    maxx = obs.index.max()

    
    p.x_range = Range1d(
        start = minx,
        end   = maxx
    )
    
    
    p.legend.location = 'top_left'

    p.add_tools(
        HoverTool(
            tooltips=[
                ('obs.', '@y'),
            ],
            renderers=[l1],
        ),
    )
    return p



#def make_plot(obs, model = None,label,remove_mean_diff=False,bbox_bias=None):
def make_plot(obs, model = None,label=None,remove_mean_diff=False,bbox_bias=0.0):    
    #TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
    TOOLS="crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,reset,save,"
    
    p = figure(toolbar_location='above',
               x_axis_type='datetime',
               width=width,
               height=height,
               tools=TOOLS)

    p.add_layout(Title(text='Station: '+obs._metadata['station_code'], text_font_style="italic"), 'above')
    p.add_layout(Title(text=obs._metadata['station_name'], text_font_size="10pt"), 'above')

    p.yaxis.axis_label = label

    obs_val = obs.values.squeeze()
    
    l1 = p.line(
        x=obs.index,
        y=obs_val,
        line_width=5,
        line_cap='round',
        line_join='round',
        legend='obs.',
        color='#0000ff',
        alpha=0.7,
    )
    

    if model is not None:
        mod_val = model.values.squeeze()
        
        if ('SSH' in label) and remove_mean_diff:
            mod_val = mod_val + obs_val.mean() - mod_val.mean()

        if ('SSH' in label) and bbox_bias is not None:
            mod_val = mod_val + bbox_bias

        l0 = p.line(
            x=model.index,
            y=mod_val,
            line_width=5,
            line_cap='round',
            line_join='round',
            legend='model',
            color='#9900cc',
            alpha=0.7,
        )
  


        minx = max (model.index.min(),obs.index.min())
        maxx = min (model.index.max(),obs.index.max())    
        
        minx = model.index.min()
        maxx = model.index.max()
    else:
        minx = obs.index.min()
        maxx = obs.index.max()

    
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


#def make_plot(obs, model = None,label,remove_mean_diff=False,bbox_bias=None):
def make_dual_plot(obs, model = None,label=None,remove_mean_diff=False,bbox_bias=0.0):    
    #TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
    
    
    if model is None:
        sys.exit('Model can not be none')
    
    
    df = obs.copy()
    df ['obs'] = obs.values
    df ['mod'] = model
    df = df.dropna()

    
    if True:
        if ('SSH' in label) and remove_mean_diff:
           df ['mod'] = df ['mod'] + df ['obs'].mean() - df ['mod'].mean()

        if ('SSH' in label) and bbox_bias is not None:
            df ['mod'] = df ['mod'] + bbox_bias    

    # https://bokeh.pydata.org/en/latest/docs/user_guide/interaction/linking.html#linked-brushing
    # create a column data source for the plots to share
    src = ColumnDataSource(data=dict(x  = df.index.to_pydatetime(), 
                                     yo = df['obs'].values,  
                                     ym = df['mod'].values))

    TOOLS="box_select,lasso_select,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,reset,save,help,"

    left = figure(toolbar_location='above',
               x_axis_type='datetime',
               width=width,
               height=height,
               tools=TOOLS)

    left.add_layout(Title(text='Station: '+obs._metadata['station_code'], text_font_style="italic"), 'above')
    left.add_layout(Title(text=obs._metadata['station_name'], text_font_size="10pt"), 'above')

    left.yaxis.axis_label = label
    left.xaxis.axis_label = 'DateTime'

    l1 = left.line(
        x='x',
        y='yo',
        line_width=5,
        line_cap='round',
        line_join='round',
        legend='obs.',
        color='#0000ff',
        alpha=0.7,
        source = src,
    )


    l0 = left.line(
        x='x',
        y='ym',
        line_width=5,
        line_cap='round',
        line_join='round',
        legend='model',
        color='#9900cc',
        alpha=0.7,
        source =src
    )



    left.x_range = Range1d(
        start = df.index.min(),
        end   = df.index.max()
    )


    left.legend.location = 'top_left'

    left.add_tools(
        HoverTool(
            tooltips=[
                ('Model', '@ym'),
            ],
            renderers=[l0],
            # display a tooltip whenever the cursor is vertically in line with a glyph
            mode='vline',            
        ),
        HoverTool(
            tooltips=[
                ('Obs.', '@yo'),
            ],
            renderers=[l1],
            # display a tooltip whenever the cursor is vertically in line with a glyph
            mode='vline',
        ),
    )
    
   # left.add_tools(
   #     HoverTool(
   #         tooltips=[
   #             ('Model', '@ym'),
   #             ('Obs.', '@yo'),
   #         ],
   #         # display a tooltip whenever the cursor is vertically in line with a glyph
   #         mode='vline'
   #     ),
   #
   # )
           
    
    
    
    
    ## display a tooltip whenever the cursor is vertically in line with a glyph
    #mode='vline'
    
    
    ####  https://bokeh.pydata.org/en/latest/docs/user_guide/tools.html
    #  https://bokeh.pydata.org/en/1.0.0/docs/user_guide/examples/tools_hover_tooltip_formatting.html
    #rTOOLTIPS = [
    #("Date" , "$x"),
    #("Obs." , "$yo"),    
    #("Model", "$ym"),
    #]
    #hover_tool.formatters = { "x": "datetime"}

    #right =  figure(tools=TOOLS, plot_width=height, plot_height=height, title='Select',tooltips=rTOOLTIPS)
    right =  figure(tools=TOOLS, plot_width=height, plot_height=height, title='Select from ...')
    
    right.circle('yo', 'ym', source=src)
    right.yaxis.axis_label = 'Model'    
    right.xaxis.axis_label = 'Obs.'    
    
    right.add_tools(
        HoverTool(
            tooltips=[
                #('Date' , '@x'),
                ('Model', '@ym'),                
                ('Obs.' , '@yo'),    
            ],
            formatters={
                'x'      : 'datetime', # use 'datetime' formatter for 'date' field
                                       # use default 'numeral' formatter for other fields
                },
        ),
        
    )
    
    
    
    
    
    p = gridplot([[right,left]])
    
    return p


#################
def make_marker(p, location, fname , color = 'green',icon= 'stats'):
    html = file_html(p, CDN, fname)
    #iframe = IFrame(html , width=width+45+height, height=height+80)
    iframe = IFrame(html , width=width * 1.1, height=height * 1.1)
    #popup = folium.Popup(iframe, max_width=2650+height)
    popup = folium.Popup(iframe)
    iconm = folium.Icon(color = color, icon=icon)
    marker = folium.Marker(location=location,
                           popup=popup,
                           icon=iconm)
    return marker

###############################
###try to add countor to map

def Read_maxele_return_plot_obj(fgrd='depth_hsofs_inp.nc',felev='maxele.63.nc'):
    """
    
    """
    
    ncg   = netCDF4.Dataset(fgrd)
    ncgv  = ncg.variables
    #read maxelev
    nc0     = netCDF4.Dataset(felev)
    ncv0    = nc0.variables
    data    = ncv0['zeta_max'][:]
    #data    = ncv0['surge'][:]
    #dep0    = ncv0['depth'][:]
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
        NumLevels = 11

    levels = np.linspace(MinVal, MaxVal, num=NumLevels)
    tri = Tri.Triangulation(lon0,lat0, triangles=elems)

    contour = plt.tricontourf(tri, data,levels=levels,cmap = my_cmap ,extend='max')
    return contour,MinVal,MaxVal,levels

#############################################################
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

#################
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

#################
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
    
#################
def get_station_wnd_all(fort61):
    """
    Read model wind
    
    """
    nc0      = netCDF4.Dataset(fort61)
    ncv0     = nc0.variables 
    try:
        sta_lon  = ncv0['x'][:]
    except:
        sta_lon  = ncv0['lon'][:]
    
    try:
        sta_lat  = ncv0['y'][:]
    except:
        sta_lon  = ncv0['lat'][:]
    
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
    
#################
def get_station_wnd(fort61):
    """
    Read model wind
    
    """
    nc0      = netCDF4.Dataset(fort61)
    ncv0     = nc0.variables 
    try:
        sta_lon  = ncv0['x'][:]
    except:
        sta_lon  = ncv0['lon'][:]
    
    try:
        sta_lat  = ncv0['y'][:]
    except:
        sta_lat  = ncv0['lat'][:]
    
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
    mod_table1 = pd.DataFrame(data = np.c_[ind, stationIDs], columns=['ind',  'station_code'])
    nc0.close()
    return mod,mod_table1


def get_model_at_station_wave(wav_at_nbdc):
    """
    Read model wave
    
    """
    nc0      = netCDF4.Dataset(wav_at_nbdc)
    ncv0     = nc0.variables 
    sta_hsig =  ncv0['hsig'] [:].squeeze()
    sta_hsig [sta_hsig > 1e3 ] = 0.0
    
    sta_date = netCDF4.num2date(ncv0['time'][:], ncv0['time'].units)
    sta_nam  = ncv0['station_name'][:].squeeze()
    sta_lon  = ncv0['lon'][:]
    sta_lat  = ncv0['lat'][:]
    nc0.close()
    
    stationIDs = []
    mod    = []
    ind = np.arange(len(sta_lat))
    for ista in ind:
        stationID = sta_nam[ista][~sta_nam.mask[ista]].tostring().decode().split('.')[0]
        stationIDs.append(stationID)
        mod_tmp = pd.DataFrame(data = np.c_[sta_date,sta_hsig[:,ista]],
                               columns = ['date_time', 'hsig' ]).set_index('date_time')
        mod_tmp._metadata = stationID
        mod.append(mod_tmp)

    stationIDs = np.array(stationIDs)
    mod_table = pd.DataFrame(data = np.c_[ind, stationIDs], columns=['ind',  'station_code'])
    return mod,mod_table


def get_model_at_station_wind(wnd_at_nbdc):
    """
    Read model wind
    
    """
    nc0      = netCDF4.Dataset(wnd_at_nbdc)
    ncv0     = nc0.variables 
    sta_wnd =  np.sqrt ( ncv0['uwnd'] [:].squeeze() ** 2 +  ncv0['vwnd']        [:].squeeze() ** 2 )
    sta_wnd [sta_wnd > 1e3 ] = 0.0
    
    sta_date = netCDF4.num2date(ncv0['time'][:], ncv0['time'].units)
    sta_nam  = ncv0['station_name'][:].squeeze()
    sta_lon  = ncv0['lon'][:]
    sta_lat  = ncv0['lat'][:]
    nc0.close()
    
    stationIDs = []
    mod    = []
    ind = np.arange(len(sta_lat))
    for ista in ind:
        stationID = sta_nam[ista][~sta_nam.mask[ista]].tostring().decode()
        stationIDs.append(stationID)
        mod_tmp = pd.DataFrame(data = np.c_[sta_date,sta_wnd[:,ista]],
                               columns = ['date_time', 'wnd' ]).set_index('date_time')
        mod_tmp._metadata = stationID
        mod.append(mod_tmp)

    stationIDs = np.array(stationIDs)
    mod_table = pd.DataFrame(data = np.c_[ind, stationIDs], columns=['ind',  'station_code'])
    return mod,mod_table

#############################################################
def make_map(bbox, **kw):
    """
    Creates a folium map instance.

    Examples
    --------
    >>> from folium import Map
    >>> bbox = [-87.40, 24.25, -74.70, 36.70]
    >>> m = make_map(bbox)
    >>> isinstance(m, Map)
    True

    """
    import folium

    line = kw.pop('line', True)
    layers = kw.pop('layers', True)
    zoom_start = kw.pop('zoom_start', 5)

    lon, lat = np.array(bbox).reshape(2, 2).mean(axis=0)
    #
    m = folium.Map(width='100%', height='100%',
                   location=[lat, lon], zoom_start=zoom_start)

    if layers:
        add = 'MapServer/tile/{z}/{y}/{x}'
        base = 'http://services.arcgisonline.com/arcgis/rest/services'
        ESRI = dict(Imagery='World_Imagery/MapServer',
                    #Ocean_Base='Ocean/World_Ocean_Base',
                    #Topo_Map='World_Topo_Map/MapServer',
                    #Physical_Map='World_Physical_Map/MapServer',
                    #Terrain_Base='World_Terrain_Base/MapServer',
                    #NatGeo_World_Map='NatGeo_World_Map/MapServer',
                    #Shaded_Relief='World_Shaded_Relief/MapServer',
                    #Ocean_Reference='Ocean/World_Ocean_Reference',
                    #Navigation_Charts='Specialty/World_Navigation_Charts',
                    #Street_Map='World_Street_Map/MapServer'
                    )

        for name, url in ESRI.items():
            url = '{}/{}/{}'.format(base, url, add)

            w = folium.TileLayer(tiles=url,
                                 name=name,
                                 attr='ESRI',
                                 overlay=False)
            w.add_to(m)

    if line:  # Create the map and add the bounding box line.
        p = folium.PolyLine(get_coordinates(bbox),
                            color='#FF0000',
                            weight=2,
                            opacity=0.5,
                            latlon=True)
        p.add_to(m)

    folium.LayerControl().add_to(m)
    return m


import cartopy.crs as ccrs
from cartopy.mpl.gridliner import (LONGITUDE_FORMATTER,
                                   LATITUDE_FORMATTER)
import cartopy.feature as cfeature 

def make_map_cartopy(projection=ccrs.PlateCarree()):                                                                                                                                        
                                                                                           
    """                                                                          
    Generate fig and ax using cartopy                                                                    
    input: projection                                                                                    
    output: fig and ax                             
    """                                  
    alpha = 0.5                                        
    subplot_kw = dict(projection=projection)                        
    fig, ax = plt.subplots(figsize=(9, 13),                           
                           subplot_kw=subplot_kw)   
    gl = ax.gridlines(draw_labels=True)                                 
    gl.xlabels_top = gl.ylabels_right = False 
    gl.xformatter = LONGITUDE_FORMATTER                        
    gl.yformatter = LATITUDE_FORMATTER                                
                                    
        # Put a background image on for nice sea rendering.             
    ax.stock_img()                                   
                                                          
    # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
    states_provinces = cfeature.NaturalEarthFeature(                      
        category='cultural',                  
        name='admin_1_states_provinces_lines',
        scale='50m',           
        facecolor='none')        

    SOURCE = 'Natural Earth'
    LICENSE = 'public domain'
                                                                                                                                                                                    
    ax.add_feature(cfeature.LAND,zorder=0,alpha=alpha)          
    ax.add_feature(cfeature.COASTLINE,zorder=1,alpha=alpha)
    ax.add_feature(cfeature.BORDERS,zorder=1,alpha=2*alpha)
                       
    ax.add_feature(states_provinces, edgecolor='gray',zorder=1)
                                                          
    # Add a text annotation for the license information to the
    # the bottom right corner.                                            
    text = AnchoredText(r'$\mathcircled{{c}}$ {}; license: {}'
                        ''.format(SOURCE, LICENSE),
                        loc=4, prop={'size': 9}, frameon=True)                                    
    ax.add_artist(text)                                                                           
                                         
    ax.set_xlim(-132,-65)  #lon limits           
    ax.set_ylim( 20 , 55)  #lat limits   
    return fig, ax


#calculate bias for bbox region
def get_bias(bias_bbox,ssh,ssh_table,fort61 ):
    ind  = get_ind (bias_bbox,ssh_table.lon,ssh_table.lat)
    mask = get_mask(bias_bbox,ssh_table.lon,ssh_table.lat)
    #
    ssh_bias_table = ssh_table[~mask]
    #
    ssh_bias = [] 
    ssh_biad_tab = []
    for in0 in ind:
        ssh_bias.append(ssh[in0])

    ########### Read SSH data
    mod , mod_table = get_station_ssh(fort61)

    ############# Sea Surface height analysis ########################
    # For simplicity we will use only the stations that have both wind speed and sea surface height and reject those that have only one or the other.
    common  = set(ssh_bias_table['station_code']).intersection(mod_table  ['station_code'].values)

    ssh_obs, ssh_mod = [], []
    for station in common:
        ssh_obs.extend([obs for obs in ssh   if obs._metadata['station_code'] == station])
        ssh_mod.extend([obm for obm in mod   if obm._metadata                 == station])


    index = pd.date_range(
        start = bias_calc_start.replace(tzinfo=None),
        end   = bias_calc_end.replace  (tzinfo=None),
        freq=freq
    )
    #############################################################
    #organize and re-index both observations
    # Re-index and rename series.
    ssh_observations = []
    ssh_all = []
    for series in ssh_obs:
        _metadata = series._metadata
        obs = series.tz_localize(None).reindex(index=index, limit=1, method='nearest')
        obs._metadata = _metadata
        obs.name = _metadata['station_name']
        ssh_observations.append(obs)
        ssh_all.append( obs['water_surface_height_above_reference_datum (m)'].values)

    ##############################################################
    #model
    ssh_from_model = []
    mod_all = []
    for series in ssh_mod:
        _metadata = series._metadata
        obs = series.tz_localize(None).reindex(index=index, limit=1, method='nearest')
        obs._metadata = _metadata
        obs.name = _metadata
        ssh_from_model.append(obs)
        mod_all.append( obs['ssh'].values)



    ssh_all = np.array(ssh_all).flatten()
    mask   = np.isnan(ssh_all)
    mod_all = np.array(mod_all).flatten()

    bias = (ssh_all[~mask] - mod_all [~mask]).mean()
    return bias
########################################
####       MAIN CODE from HERE     #####
########################################

for key in storms.keys():
    name = storms[key]['name']
    year = storms[key]['year']

    print('\n\n\n\n\n\n********************************************************')
    print(            '*****  Storm name ',name, '      Year ',  year, '    *********')
    print(            '******************************************************** \n\n\n\n\n\n')


    wnd_ocn_observs = wnd_ocn_models = wnd_ocn = None
    wav_ocn = wav_observs = wav_models = None
    wnd_obs = wnd_observs = wnd_models = None
    ssh = ssh_observations = ssh_from_model = None


    name  = name  #[:3]
    obs_dir = os.path.join(base_dirf,'work_dir','obs')
    mod_dir = os.path.join(base_dirf,'work_dir','mod')


    print ( ' > Read NHC information ... ')
    al_code , hurricane_gis_files = get_nhc_storm_info (year,name)

    #donload gis zip files
    base = download_nhc_gis_files(hurricane_gis_files)

    # get advisory cones and track points
    cones , points , pts = read_advisory_cones_info(hurricane_gis_files,base,year,al_code)

    # Find the bounding box to search the data.
    bbox_from_track = True
    if bbox_from_track:
        last_cone = cones[-1]['geometry'].iloc[0]
        track = LineString([point['geometry'] for point in points])
        track_lons = track.coords.xy[0]
        track_lats = track.coords.xy[1]
        bbox = min(track_lons)-2, min(track_lats)-2, max(track_lons)+2, max(track_lats)+2
    else:
        bounds = np.array([last_cone.buffer(2).bounds, track.buffer(2).bounds]).reshape(4, 2)
        lons, lats = bounds[:, 0], bounds[:, 1]
        bbox = lons.min(), lats.min(), lons.max(), lats.max()

    if storms[key]['bbox'] is not None:
        bbox = storms[key]['bbox']
    
    # Note that the bounding box is derived from the track and the latest prediction cone.
    strbbox = ', '.join(format(v, '.2f') for v in bbox)

    #######################################
    print('\n\n\n\n\n\n********************************************************')
    print(            '*****  Storm name ',name, '      Year ',  year, '    *********')
    print(            '******************************************************** \n\n\n\n\n\n')

    #######################################

    fgrd      = os.path.join(mod_dir,'depth_hsofs_inp.nc')
    fhwm      = os.path.join(obs_dir,'hwm/' + (name+year).lower() + '.csv')

    dirs = np.array(glob(os.path.join(mod_dir,name)+'/*'))
    if len(dirs) == 0:
        dirs = [os.getcwd()+'/']
        print ('WARNING: Incorrect Model directory or not model results ready yet!')

    for dir0 in dirs[:]:
        fort61       = dir0 + '/fort_wind.61.nc'
        wav_at_nbdc  = dir0 + '/01_wave_on_ndbc_obs.nc'
        wnd_at_nbdc  = dir0 + '/01_wind_on_ndbc_obs.nc'
        felev        = dir0 + '/maxele.63_all.nc'
        ###################################
        
        try:
            forcing   =  dir0.split('/')[-1].split('.')[1]
            key0 =  dir0.split('/')[-1].split('.')[0]
        except:
            forcing   =  '_'
            key0      =  '_'
            pass
                    
        print ( ' > ', dir0)
        print ( ' > \n\n\n storm:', name, '  Year:', year,'   HWRF case:',forcing,'   Case:', key0, '\n\n\n') 
        
        #year = '2012'
        #name = 'SANDY'

        #####################################
        # Now we can get all the information we need from those GIS files. Let's start with the event dates.
        ######################################

        # We are ignoring the timezone, like AST (Atlantic Time Standar) b/c
        # those are not a unique identifiers and we cannot disambiguate.

        if  'FLDATELBL' in points[0].keys():
            start = points[0]['FLDATELBL']
            end = points[-1]['FLDATELBL']
            date_key = 'FLDATELBL'

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
        print ( ' > bbox: {}\nstart: {}\n  end: {}'.format(strbbox, start, end))
        start_dt = start_dt  - obs_xtra_days
        end_dt   = end_dt    + obs_xtra_days

        ############################################################
        # read data

        print (' read from CSV files')
        try:
            print ('   >  ssh from CSV files')
            ssh_table,ssh          = read_csv (obs_dir, name, year, label='coops_ssh' )
        except:
            print ('   >  No ssh CSV files')            

        try:
            print ('   >  wind from CSV files')
            wnd_obs_table,wnd_obs  = read_csv (obs_dir, name, year, label='coops_wind')
        except:
            print ('   >>  No wind CSV files')  

        try:
            print ('   >  NDBC wind from CSV files')
            wnd_ocn_table,wnd_ocn  = read_csv (obs_dir, name, year, label='ndbc_wind' )
        except:
            print ('   >>  No ndbc wind CSV files')  

        try:
            print ('   >  NDBC wave from CSV files')
            wav_ocn_table, wav_ocn = read_csv (obs_dir, name, year, label='ndbc_wave' )
        except:
            print ('   >> No  NDBC wave CSV files')  

        freq = '15min'
        freq = '30min'

        print (' >> re-index time series to every : {}'.format(freq))

        ############################################################
        try:
            ########### Read SSH data
            mod , mod_table = get_station_ssh(fort61)
            
            start_dt = max( start_dt.replace(tzinfo=None) , pd.to_datetime(mod[0].index[0]))
            end_dt   = min( end_dt.replace(tzinfo=None)   , pd.to_datetime(mod[0].index[-1]))            
            ############# Sea Surface height analysis ########################
            # For simplicity we will use only the stations that have both wind speed and sea surface height and reject those that have only one or the other.
            common  = set(ssh_table['station_code']).intersection(mod_table  ['station_code'].values)

            ssh_obs, ssh_mod = [], []
            for station in common:
                ssh_obs.extend([obs for obs in ssh   if obs._metadata['station_code'] == station])
                ssh_mod.extend([obm for obm in mod   if obm._metadata                 == station])

            index = pd.date_range(
                start = start_dt.replace(tzinfo=None),
                end   = end_dt.replace  (tzinfo=None),
                freq=freq
            )
            index = index.tz_localize(None) 
            #############################################################
            #organize and re-index both observations
            # Re-index and rename series.
            ssh_observations = []
            for series in ssh_obs:
                #series = series.tz_localize(None)
                _metadata = series._metadata
                obs = series.tz_localize(None).reindex(index=index, limit=1, method='nearest')
                obs._metadata = _metadata
                obs.dropna(inplace=True)
                obs.name = _metadata['station_name']
                ssh_observations.append(obs)

            ##############################################################
            #model
            ssh_from_model = []
            for series in ssh_mod:
                _metadata = series._metadata
                #series = series.tz_localize(None)
                mod0 = series.tz_localize(None).reindex(index=index, limit=1, method='nearest')
                mod0._metadata = _metadata
                mod0.name = _metadata
                mod0['ssh'][np.abs(mod0['ssh']) > 10] = np.nan
                mod0.dropna(inplace=True)
                ssh_from_model.append(mod0)
        
            
            if apply_bbox_bias:
                bbox_bias =  get_bias(bias_bbox,ssh,ssh_table,fort61)
                print ( ' Bbox bias correction   bias= {}'.format(bbox_bias) )
            else:
            	bbox_bias = 0.0
       
            ssh_coops_stations = True
        except:
            print (' WARNING >>>  fort.63.nc SSH is not available ..')
            ssh_coops_stations = False
        ############# Wind COOPS and model analysis ########################
        try:
            #read wind model data
            wnd_mod , wnd_mod_table = get_station_wnd(fort61)

            # For simplicity we will use only the stations that have both wind speed and sea surface height and reject those that have only one or the other.
            commonw  = set(wnd_obs_table['station_code']).intersection(wnd_mod_table  ['station_code'].values)

            wobs, wmod = [], []
            for station in commonw:
                wobs.extend([obs for obs in wnd_obs   if obs._metadata['station_code'] == station])
                wmod.extend([obm for obm in wnd_mod   if obm._metadata                 == station])


            index = pd.date_range(
                start = start_dt.replace(tzinfo=None),
                end   = end_dt.replace  (tzinfo=None),
                freq=freq
            )
            #############################################################
            #organize and re-index both observations
            # Re-index and rename series.
            wnd_observs = []
            for series in wobs:
                _metadata = series._metadata
                series = series.tz_localize(None)
                obs = series.tz_localize(None).reindex(index=index, limit=1, method='nearest')
                obs._metadata = _metadata
                obs.name = _metadata['station_name']
                wnd_observs.append(obs)

            ##############################################################
            #model
            wnd_models = []
            for series in wmod:
                _metadata = series._metadata
                #series = series.tz_localize(None)
                obs = series.tz_localize(None).reindex(index=index, limit=1, method='nearest')
                obs._metadata = _metadata
                obs.name = _metadata
                #obs['ssh'][np.abs(obs['ssh']) > 10] = np.nan
                obs.dropna(inplace=True)
                wnd_models.append(obs)
            wind_coops_stations = True  
        except:
            print (' WARNING >>>  fort.61 does not include wind info ..')
            wind_coops_stations = False


        ############# Wave NDBC obs and model analysis ########################
        try:
            #read wave model data
            wav_mod , wav_mod_table = get_model_at_station_wave(wav_at_nbdc)
            commonwav  = set(wav_ocn_table['station_code']).intersection(wav_mod_table['station_code'].values)

            wav_ocns, wav_mods = [], []
            for station in commonwav:
                wav_ocns.extend([obs for obs in wav_ocn   if obs._metadata['station_code'] == station])
                wav_mods.extend([obm for obm in wav_mod   if obm._metadata                 == station])

            index = pd.date_range(
                start = start_dt.replace(tzinfo=None),
                end   = end_dt.replace  (tzinfo=None),
                freq=freq
            )
            #############################################################
            #organize and re-index both observations
            # Re-index and rename series.
            wav_observs = []
            for series in wav_ocns:
                _metadata = series._metadata
                series = series.tz_localize(None)
                obs = series.tz_localize(None).reindex(index=index, limit=1, method='nearest')
                obs._metadata = _metadata
                obs.name = _metadata['station_name']
                wav_observs.append(obs)

            ##############################################################
            #model
            wav_models = []
            for series in wav_mods:
                _metadata = series._metadata
                #series = series.tz_localize(None)
                obs = series.tz_localize(None).reindex(index=index, limit=1, method='nearest')
                obs._metadata = _metadata
                obs.name = _metadata
                #obs['ssh'][np.abs(obs['ssh']) > 10] = np.nan
                obs.dropna(inplace=True)
                wav_models.append(obs)
            wave_ndbc_stations = True  
        except:
            print (' WARNING >>>  not include wave info ..')
            wave_ndbc_stations = False

        ############# wind NDBC obs and model analysis ########################
        try:
            #read wind model data
            wnd_ocn_mod,wnd_ocn_mod_table = get_model_at_station_wind(wnd_at_nbdc)
            
            commonwnd  = set(wnd_ocn_table['station_code']).intersection(wnd_ocn_mod_table['station_code'].values)

            wnd_ocns, wnd_ocn_mods = [], []
            for station in commonwnd:
                wnd_ocns.extend    ([obs for obs in wnd_ocn   if obs._metadata['station_code'] == station])
                wnd_ocn_mods.extend([obm for obm in wnd_ocn_mod   if obm._metadata                 == station])

            index = pd.date_range(
                start = start_dt.replace(tzinfo=None),
                end   = end_dt.replace  (tzinfo=None),
                freq=freq
            )
            #############################################################
            #organize and re-index both observations
            # Re-index and rename series.
            wnd_ocn_observs = []
            for series in wnd_ocns:
                _metadata = series._metadata
                series = series.tz_localize(None)
                obs = series.tz_localize(None).reindex(index=index, limit=1, method='nearest')
                obs._metadata = _metadata
                obs.name = _metadata['station_name']
                wnd_ocn_observs.append(obs)

            ##############################################################
            #model
            wnd_ocn_models = []
            for series in wnd_ocn_mods:
                _metadata = series._metadata
                #series = series.tz_localize(None)
                obs = series.tz_localize(None).reindex(index=index, limit=1, method='nearest')
                obs._metadata = _metadata
                obs.name = _metadata
                #obs['ssh'][np.abs(obs['ssh']) > 10] = np.nan
                obs.dropna(inplace=True)
                wnd_ocn_models.append(obs)
            wind_ndbc_stations = True  
        except:
            print (' WARNING >>>  not include wind info ..')
            wind_ndbc_stations = False


        #######################################################
        print('  > Put together the final map')


        if False:
            m = make_map(bbox)
        else:

            # Here is the final result. Explore the map by clicking on the map features plotted!
            lon = track.centroid.x
            lat = track.centroid.y
            ############################################################
            #if  'FLDATELBL' in points[0].keys():
            ##
            m = folium.Map(location=[lat, lon], tiles='OpenStreetMap', zoom_start=4, control_scale=True)
            Fullscreen(position='topright', force_separate_button=True).add_to(m)


            add  = 'MapServer/tile/{z}/{y}/{x}'
            base = 'http://services.arcgisonline.com/arcgis/rest/services'
            ESRI = dict(Imagery='World_Imagery/MapServer')

            for name1, url in ESRI.items():
                url = '{}/{}/{}'.format(base, url, add)

                w = folium.TileLayer(tiles=url,
                                     name=name1,
                                     attr='ESRI',
                                     overlay=False)
                w.add_to(m)

        #################################################################
        try:
            print('     > Plot max water elev ..')
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
        except:
            print('      > WARNING: ADCIRC maxelev in not available!')
        #
        #################################################################
        #try:
        print('     > Plot COOPS SSH stations ..')
        #marker_cluster_coops = MarkerCluster(name='CO-OPS SSH observations')
        marker_cluster_coops = MarkerCluster(name='CO-OPS observations')
        marker_cluster_coops.add_to(m)    
        
        if ssh_coops_stations:
            # ssh Observations stations 
            for ssh1, model1 in zip(ssh_observations, ssh_from_model):
                fname = ssh1._metadata['station_code']
                location = ssh1._metadata['lat'], ssh1._metadata['lon']
                ###p = make_plot(ssh1, model1, label='SSH [m]',remove_mean_diff=remove_mean_diff, bbox_bias = bbox_bias)
                
                p = make_dual_plot(ssh1, model1, label='SSH [m]',remove_mean_diff=remove_mean_diff, bbox_bias = bbox_bias)


                #p = make_plot(ssh1, ssh1)    
                marker = make_marker(p, location=location, fname=fname)
                marker.add_to(marker_cluster_coops)
                #del ssh_observations, ssh_from_model
        else:
            print ('      > plot coops obs only')
            for ssh1 in ssh:
                fname = ssh1._metadata['station_code']
                location = ssh1._metadata['lat'], ssh1._metadata['lon']
                p = make_plot_obs(ssh1, label='SSH [m]')
                #p = make_plot(ssh1, ssh1)    
                marker = make_marker(p, location=location, fname=fname)
                marker.add_to(marker_cluster_coops)
                #
                #del ssh
        #except:
        #    print('      > WARNING: CO-OPS SSH OBS in not available!')
          
        ################################################################
        try:
            print('     > Plot COOPS wind stations ..')
            #marker_clusterw = MarkerCluster(name='CO-OPS wind observations')
            marker_clusterw = marker_cluster_coops
            marker_clusterw.add_to(m)
            if wind_coops_stations:
                # Wind Observations stations.
                for ssh1, model1 in zip(wnd_observs,wnd_models):
                    fname = ssh1._metadata['station_code']
                    location = ssh1._metadata['lat'] , ssh1._metadata['lon'] 
                    p = make_plot(ssh1, model1,'Wind [m/s]')
                    #p = make_plot(ssh1, ssh1)    
                    marker = make_marker(p, location=location, fname=fname, color = 'gray',icon='flag')
                    marker.add_to(marker_clusterw)
                    #del wnd_observs,wnd_models
            else: 
                # Only Wind Observations.
                for ssh1 in wnd_obs:
                    fname = ssh1._metadata['station_code']
                    location = ssh1._metadata['lat'] , ssh1._metadata['lon'] 
                    p = make_plot_obs(ssh1,'Wind [m/s]')
                    #p = make_plot(ssh1, ssh1)    
                    marker = make_marker(p, location=location, fname=fname, color = 'gray',icon='flag')
                    marker.add_to(marker_clusterw)
                    #del wnd_obs
        except:
            print('      > WARNING: COOPS wind OBS in not available!')

        #################################################################
        try:
            print('     > Plot NDBC wave stations ..')
            #marker_cluster_ndbc = MarkerCluster(name='NDBC wave observations')
            marker_cluster_ndbc = MarkerCluster(name='NDBC observations')
            marker_cluster_ndbc.add_to(m)

            if wave_ndbc_stations:
                for ssh1, model1 in zip(wav_observs,wav_models):
                    
                    obs = ssh1['sea_surface_wave_significant_height (m)']
                    obs._metadata = ssh1._metadata
                    
                    fname = ssh1._metadata['station_code']
                    location = ssh1._metadata['lat'] , ssh1._metadata['lon'] 
                    p = make_plot(obs, model1,'Hsig [m]')
                    #p = make_plot(ssh1, ssh1)    
                    marker = make_marker(p, location=location, fname=fname, color = 'darkpurple',icon='record')
                    marker.add_to(marker_cluster_ndbc)
                    #del wav_observs,wav_models
            else:
                for ssh1 in wav_ocn:
                    obs = ssh1['sea_surface_wave_significant_height (m)']
                    obs._metadata = ssh1._metadata
                    
                    fname = ssh1._metadata['station_code']
                    location = ssh1._metadata['lat'] , ssh1._metadata['lon'] 
                    p = make_plot_obs(obs,'Hsig [m]')
                    #p = make_plot(ssh1, ssh1)    
                    marker = make_marker(p, location=location, fname=fname, color = 'darkpurple',icon='record')
                    marker.add_to(marker_cluster_ndbc)
                    #del wav_ocn
        except:
            print('      > WARNING: NDBS wave OBS in not available!')
        ########################################################################
        try:
            print('     > Plot NDBC wind stations ..')
            #marker_cluster_ndbcw = MarkerCluster(name='NDBC wind observations')
            marker_cluster_ndbcw = marker_cluster_ndbc
            
            marker_cluster_ndbcw.add_to(m)

            if wind_ndbc_stations:
                for ssh1, model1 in zip(wnd_ocn_observs,wnd_ocn_models):
                    obs = ssh1['wind_speed (m/s)']
                    obs._metadata = ssh1._metadata
                    fname = ssh1._metadata['station_code']
                    location = ssh1._metadata['lat'] , ssh1._metadata['lon'] 
                    p = make_plot(obs, model1,'Wind [m/s]')
                    #p = make_plot(ssh1, ssh1)    
                    marker = make_marker(p, location=location, fname=fname, color = 'orange',icon='flag')
                    marker.add_to(marker_cluster_ndbc)
                    #del wnd_ocn_observs,wnd_ocn_models
            else:
                for ssh1 in wnd_ocn:
                    obs = ssh1['wind_speed (m/s)']
                    obs._metadata = ssh1._metadata
                    fname = ssh1._metadata['station_code']
                    location = ssh1._metadata['lat'] , ssh1._metadata['lon'] 
                    p = make_plot_obs(obs,'Wind [m/s]')
                    #p = make_plot(ssh1, ssh1)    
                    marker = make_marker(p, location=location, fname=fname, color = 'orange',icon='flag')
                    marker.add_to(marker_cluster_ndbcw)
                    #del wnd_ocn
        except:
            print('      > WARNING: NDBS wind OBS in not available!')        
        ###################        
        ## Plotting bounding box
        # folium.LayerControl().add_to(m)
        p = folium.PolyLine(get_coordinates(bbox),
                            color='#009933',
                            weight=2,
                            opacity=0.6)

        p.add_to(m)
        ##################################################### 

        track_radius = 5
        if plot_cones:
            print('     > Plot NHC cone predictions')
            marker_cluster1 = MarkerCluster(name='NHC cone predictions')
            marker_cluster1.add_to(m)
            def style_function_latest_cone(feature):
                return {
                    'fillOpacity': 0,
                    'color': 'red',
                    'stroke': 1,
                    'weight': 0.3,
                    'opacity': 0.5,
                }

            def style_function_cones(feature):
                return {
                    'fillOpacity': 0,
                    'color': 'lightblue',
                    'stroke': 1,
                    'weight': 0.3,
                    'opacity': 0.5,
                }    
            
            

            
            if False:
                # Latest cone prediction.
                latest = cones[-1]
                ###
                if  'FLDATELBL' in points[0].keys():    #Newer storms have this information
                    names3 = 'Cone prediction as of {}'.format(latest['ADVDATE'].values[0])
                else:
                    names3 = 'Cone prediction'
                ###
                folium.GeoJson(
                    data=latest.__geo_interface__,
                    name=names3,            
                    style_function=style_function_cones,
                ).add_to(m)
            ###
            # Past cone predictions.
            for cone in cones[:-1]:
                folium.GeoJson(
                    data=cone.__geo_interface__,
                    style_function=style_function_latest_cone,
                ).add_to(marker_cluster1)

            # Latest points prediction.
            for k, row in pts.iterrows():
                
                if  'FLDATELBL' in points[0].keys():    #Newer storms have this information
                    date = row[date_key] 
                    hclass = row['TCDVLP']
                    popup = '{}<br>{}'.format(date, hclass)
                    if 'tropical' in hclass.lower():
                        hclass = 'tropical depression'
        
                    color = colors_hurricane_condition [hclass.lower()]
                else:
                    popup = '{}<br>{}'.format(name,year )
                    color = colors_hurricane_condition ['hurricane']
              
                location = row['LAT'], row['LON']
                folium.CircleMarker(
                    location=location,
                    radius=track_radius,
                    fill=True,
                    color=color,
                    popup=popup,
                ).add_to(m)
        ####################################################
    
    print('     > Plot points along the final track ..')
    #marker_cluster3 = MarkerCluster(name='Track')
    #marker_cluster3.add_to(m)

    for point in points:
        if  'FLDATELBL' in points[0].keys():    #Newer storms have this information
            date = point[date_key]
            hclass = point['TCDVLP']
            popup = """
                <div style="width: 200px; height: 90px" </div>
                <h5> {} condition</h5> <br>
                'Date:      {} <br> 
                 Condition: {} <br>
                """.format(name, date, hclass)
            
            if 'tropical' in hclass.lower():
                hclass = 'tropical depression'
            
            color = colors_hurricane_condition [hclass.lower()]
        else:
            popup = '{}<br>{}'.format(name,year )
            color = colors_hurricane_condition ['hurricane']
        

        
        
        
        
        
        
        location = point['LAT'], point['LON']
        folium.CircleMarker(
            location=location,
            radius=track_radius,
            fill=True,
            color=color,
            popup=popup,
        ).add_to(m)

    #m = make_map(bbox)
    ####################################################
    try:
        print('     > Plot High Water Marks ..')
        df = pd.read_csv(fhwm)
        lon_hwm = df.longitude.values
        lat_hwm = df.latitude.values
        hwm     = df.elev_m.values
        
        hwm_ground_grp = folium.FeatureGroup(name= 'USGS High water marks')
        
        # HWMs
        for im in range (len(hwm)):
            location = lat_hwm[im], lon_hwm[im]

            words = ' '.join(df['hwmQualityName'][im].split()[:3]) +'\t'+ ' '.join(df['hwmTypeName'][im].split()[:3])
            

            popup = """
            <div style="width: 180px;" </div>
            <h4> USGS High Water Mark data</h4> <br>
            'Site ID:     {} <br> 
            Description:  {} <br>
            Elev:         {} [m]'
            """.format(df['site_id'][im],words,str(hwm[im])[:4])
            

            try:
                ind = np.argmin (np.abs(levels-hwm[im]))
                colorc = colors_elev[ind]
            except:
                colors_new = ['#d7191c',  '#fdae61',  '#ffffbf',  '#abdda4',  '#2b83ba']
                colorc =  'darkred'
            
            
            folium.CircleMarker(
                location=location,
                radius = 3,
                fill   = True,
                fill_color = colorc,
                fill_opacity = 75,
                color        = colorc,
                popup        = popup,
                ).add_to(hwm_ground_grp)
                
            hwm_ground_grp.add_to(m)
    except:
        print('WARNING: High Water Mark Data not available!')
    #################################################
    print ('     > Add disclaimer and storm name ..')
    noaa_logo = 'https://www.nauticalcharts.noaa.gov/images/noaa-logo-no-ring-70.png'
    #FloatImage(noaa_logo, bottom=90, left=5).add_to(m)    #in percent


    storm_info_html ='''
                    <div style="position: fixed; 
                                bottom: 50px; left: 5px; width: 140px; height: 45px; 
                                border:2px solid grey; z-index:9999; font-size:14px;background-color: lightgray;opacity: 0.9;
                                ">&nbsp; Storm: {} <br>
                                  &nbsp; Year:  {}  &nbsp; <br>
                    </div>
                    '''.format(name,year) 

    m.get_root().html.add_child(folium.Element(storm_info_html))
    ############################################################################
    Disclaimer_html =   '''
                    <div style="position: fixed; 
                                bottom: 5px; left: 250px; width: 520px; height: px; 
                                border:2px solid grey; z-index:9999; font-size:12px; background-color: lightblue;opacity: 0.6;
                                ">&nbsp; Hurricane Explorer;  
                                <a href="https://nauticalcharts.noaa.gov/" target="_blank" >         NOAA/NOS/OCS</a> <br>
                                  &nbsp; Contact: Saeed.Moghimi@noaa.gov &nbsp; <br>
                                  &nbsp; Disclaimer: Experimental product. All configurations and results are pre-decisional and for official use only.<br>
                    </div>
                    ''' 

    m.get_root().html.add_child(folium.Element(Disclaimer_html))
    ###################################################
    folium.LayerControl().add_to(m)
    MousePosition().add_to(m)

    print ('     > Save file ...')
    
    outh = './html_out/'
    if not os.path.exists(outh):
        os.makedirs(outh)
    
    fname = './html_out/z_{}_{}_{}_{}_storm.html'.format(year,name.split()[-1].lower(),forcing,key0)
    m.save(fname)




