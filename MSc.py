# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 08:41:49 2021

@author: Owen O'Driscoll
"""

# %reset -f

import numpy as np  
from matplotlib import pyplot as plt  
import os  
import netCDF4 as nc
import cv2
from scipy import ndimage
os.chdir('C:/Python/AAA_GRS2/mscthesis')

import MSc_equations as msc

#%% load SAR data

# path is the name of the relevant NRCS.nc file from snap
path = 'NRCS.nc'
netcdf_file = nc.Dataset(path, 'r', format='NETCDF4')
# print(netcdf_file)

# load backscatter, incidence angle and lat lon
Sigma0_VV_original_preclip = netcdf_file['Sigma0_VV'][:]
longitude = np.array(netcdf_file['lon'][:])
latitude = np.array(netcdf_file['lat'][:])
incident_angle = netcdf_file['incident_angle'][:]

# compute sample rate and range/azimuth look angle
pi = 3.1415926535
samplerate = msc.sampleRate(longitude,latitude)
rangeLookAngle = msc.calculate_initial_compass_bearing((latitude[0,0],longitude[0,0]),(latitude[0,-1],longitude[0,-1]))
azimuthAngle = rangeLookAngle - 90

if azimuthAngle <=0:
    azimuthAngle+=360

# load and visualise entire radar image    
Sigma0_VV_original = np.array(Sigma0_VV_original_preclip)
incident_original = np.array(incident_angle)
msc.plottings(Sigma0_VV_original, samplerate[1], title = 'NRCS  $\sigma_0$')


#%% apply custom low pass filter to image if desired (for instance if significant unwante high frequency signals are present)

# Sigma0_VV = msc.lowpass(Sigma0_VV, 51, samplerate[1]*3, samplerate[1], cmap = 'Greys_r', plotting = True)
# Sigma0_VV_original = msc.lowpass(Sigma0_VV_original, 51, samplerate[1]*12, samplerate[1], cmap = 'Greys_r', plotting = True)

#%% load ECMWF data

# give name of ecmwf.nc file from S-1 OCN product containing only wind speed field and wind direction field
path = 'ecmwf_carolina.nc' 
netcdf_file = nc.Dataset(path, 'r', format='NETCDF4')
# print(netcdf_file)

wind_ecmwf = netcdf_file['vv_001_owiEcmwfWindSpeed'][:]
direction_ecmwf = np.array(netcdf_file['vv_001_owiEcmwfWindDirection'][:])
direction_ecmwf_median, direction_MAD_ecmwf  = np.median(direction_ecmwf), msc.MAD(direction_ecmwf)
longitude_wind = np.array(netcdf_file['lon'][:])
latitude_wind = np.array(netcdf_file['lat'][:])
samplerate_wind = msc.sampleRate(longitude_wind,latitude_wind)
windspeed3 = np.array(wind_ecmwf)
windspeed3 = windspeed3[:,windspeed3[0,:]!=-999]

# plot wind speed field and directions
msc.plottings(windspeed3, samplerate_wind[1], title = 'OCN windspeed data [m/s]', cmap = 'jet')
msc.plottings(np.array(netcdf_file['vv_001_owiEcmwfWindDirection'][:]), samplerate_wind[1], title = 'ECMWF wind direction [deg]', cmap = 'Reds')

# print wind field statistics
print(r'ECMWF Wind origin w.r.t. North: %1.1f degrees' %direction_ecmwf_median + '+- %1.1f degrees' %direction_MAD_ecmwf)
print(r'ECMWF Wind speed: %1.1f m/s' %np.mean(windspeed3) + '+- %1.1f m/s' %np.std(windspeed3))

#%% tiled calculations 

True_wind = None # determine wind direction manually (None) or input a-priori mean value or an array of wind directions equal 
                 # in size to that of the number of tiles (e.g. direction_ecmwf). Requires to load relevant ecmwf.nc file containing
                 # wind direction field, as performed below
Zi = None  # determine Zi manually (None) or input a-priori value of Zi
size = int(83*1)    # size of tile, standard is 83 pixels (83 pixels * 300 m is approx 25km)
w_star_threshold = 0.15 # inertial subrange normalised standard deviation threshold 
dissip_rate =  1.0  # near-surface dissipation rate for loop 2B, values range between 0.5 and 2.6, see Kaimal et al., 1976
slope_multiplier = 1.0  # slope multiplier of mean wind field, multiplier of 1 means no multiplication
form = 'rolls' # convection form, either cells or rolls

wind_origin, mean_lon, mean_lat, u_star, z_0, sigma_u1, sigma_u2, L1, L2, L3, w_star2, w_star2_std, Zi_estimated,\
    corr_fact1, corr_fact2, windspeed, tiles_lon, tiles_lat, idxKeep_tiles, hold_prediction, hold_form, epsilon,\
        H, dissip_rate_loop2B \
    = msc.tiledWind(Sigma0_VV_original, incident_original, longitude, latitude, iterations = 15, size = size, \
                    dissip_rate = dissip_rate, slope_multiplier = slope_multiplier, w_star_threshold = w_star_threshold, \
                        Zi_input = Zi, True_wind = True_wind, form = form, samplerate = samplerate[1], plotting = False)


# sets errorneous wind direction of 0 to nan
epsilon2 =  sigma_u2 / sigma_u2 * epsilon
wind_origin2 = sigma_u2 / sigma_u2 * wind_origin
hold_prediction = sigma_u2 / sigma_u2 * hold_prediction  
hold_form = sigma_u2 / sigma_u2 * hold_form

# plot results
figFinal, pltsFinal = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), sharex = False, sharey = False)

bins = list(np.arange(0, 361, 15))
im1 = pltsFinal[0,0].hist(wind_origin2.flatten(), bins=bins, density=False,  alpha = 0.6, color= 'r', ls='dotted')
pltsFinal[0,0].set_title('Wind direction  [deg]', fontsize = 15)

bins = list(np.arange(150, 2050, 150))
im2 = pltsFinal[0,1].hist(Zi_estimated.flatten(), bins=bins, density=False,  alpha = 0.6, color= 'g', ls='dotted')
pltsFinal[0,1].set_title('$Z_i$  [m]', fontsize = 15)

bins = list(np.arange(0.1, 1.5, 0.02))
im2 = pltsFinal[0,2].hist(sigma_u2.flatten(), bins=bins, density=False,  alpha = 0.6, color= 'b', ls='dotted')
pltsFinal[0,2].set_title(r'$\sigma_u$  [m/s]', fontsize = 15)

bins = np.logspace(np.log10(1),np.log10(10000), 50)
im2 = pltsFinal[1,0].hist(-L2[np.isnan(L2)==False].flatten(), bins = bins, density=False,  alpha = 0.6, color= 'k', ls='dotted')
pltsFinal[1,0].set_title(r'$-L$  [m]', fontsize = 15)
pltsFinal[1,0].semilogx()

bins = list(np.arange(0, 1.0, 0.02))
im2 = pltsFinal[1,1].hist(u_star.flatten(), bins=bins, density=False,  alpha = 0.8, color= 'C04', ls='dotted')
pltsFinal[1,1].set_title(r'$u_*$  [m/s]', fontsize = 15)

bins = list(np.arange(0.5, 15.5, 0.5))
im2 = pltsFinal[1,2].hist(windspeed.flatten(), bins=bins, density=False,  alpha = 0.8, color= 'C08', ls='dotted')
pltsFinal[1,2].set_title(r'$\overline{U_n}$  [m/s]', fontsize = 15)
figFinal.subplots_adjust(hspace = 0.4, wspace =0.3)

figFinal.text(0.07, 0.5, 'Occurence [-]', ha='center', va='center', rotation='vertical', fontsize=15)

plt.figure()
bins = list(np.arange(0, 0.3, 0.001))
plt.hist(w_star2_std.flatten(), bins=bins, density=False,  alpha = 0.6, color= 'k', ls='dotted')


# print some result statistics
tiles_left = np.size(wind_origin2)-sum(np.isnan(wind_origin2))
tiles_classified = np.size(Zi_estimated)-sum(np.isnan(Zi_estimated))
print('Tiles: %1.1i ' %np.size(L2))
print('Tiles left: %1.1i ' %tiles_left)
print('wind direction: %1.1f ' %np.nanmedian(wind_origin2) + '$\pm$ %1.1f' %msc.MAD(wind_origin2))
print('wind speed: %1.1f ' %np.nanmedian(windspeed) + '$\pm$ %1.1f' %msc.MAD(windspeed))
print('Zi: %1.1i ' %np.nanmedian(Zi_estimated) + '$\pm$ %1.1i' %msc.MAD(Zi_estimated))
print('u_star: %1.3f ' %np.nanmedian(u_star) + '$\pm$ %1.3f' %msc.MAD(u_star)) 
print('sigma_u2: %1.3f ' %np.nanmedian(sigma_u2) + '$\pm$ %1.3f' %msc.MAD(sigma_u2))
print('L2: %1.1i ' %np.nanmedian(L2) + '$\pm$ %1.1i' %msc.MAD(L2))
print('H: %1.3f ' %np.nanmedian(H) + '$\pm$ %1.3f' %msc.MAD(H))
print('Form: %1.3f ' %(np.nansum(hold_form)/tiles_classified))
print('Prediction: %1.2f ' %np.nanmean(hold_prediction) + '$\pm$ %1.2f' %np.nanstd(hold_prediction))
print('ratio: %1.3f' %np.nanmedian(sigma_u2/u_star))

#%% optionally plot wind direction on map

import cartopy
import cartopy.crs as ccrs

angles =np.where(wind_origin2>=180,wind_origin2-180,wind_origin2 )
 
mean_total_lon = np.mean(longitude)
mean_total_lat = np.mean(latitude)

plus = 1.5
extent = [mean_total_lon-plus, mean_total_lon+plus, mean_total_lat-plus, mean_total_lat+plus]

def vectordir(wind_origin,rangeLookAngle):
    u = np.cos(wind_origin*pi/180)
    v = np.sin(wind_origin*pi/180)
    return u, v

angles2 = np.where(angles>=0,angles-180,angles)
u, v = vectordir(angles2,rangeLookAngle )
crs = ccrs.RotatedPole(pole_longitude=180, pole_latitude=90)


fig, ax = plt.subplots(figsize=(15,12))

ax = plt.axes(projection=ccrs.Orthographic(mean_total_lon, mean_total_lat))
ax.add_feature(cartopy.feature.OCEAN, zorder=0)
ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black')
ax.set_extent(extent)
ax.gridlines()

msc.scale_bar(ax, 100)

ax.quiver(mean_lon, mean_lat, v, u, scale=14, transform = crs)
plt.show()

#%% optionally plot other variables on map

import cartopy
import cartopy.crs as ccrs
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

# determine domain on world map
plus = 2.5
mean_total_lon = np.mean(mean_lon)
mean_total_lat = np.mean(mean_lat)
extent = [mean_total_lon-plus, mean_total_lon+plus, mean_total_lat-plus, mean_total_lat+plus]
crs = ccrs.RotatedPole(pole_longitude=180, pole_latitude=90)

# print on domain 
fig, ax = plt.subplots(figsize=(15,12))

ax = plt.axes(projection=ccrs.Orthographic(mean_total_lon, mean_total_lat))
ax.add_feature(cartopy.feature.OCEAN, zorder=0)
ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black')
ax.set_extent(extent)

# variables which can be printed
# u_star, z_0, sigma_u1, sigma_u2, L1, L2, w_star2, w_star2_std, Zi_estimated, hold_prediction, 
# hold_form, windspeed, epsilon, H

# select variable
variable = L2

im = Sigma0_VV_original_preclip
M, N = np.array(np.shape(im))//size
test2 = np.reshape(variable, (M,N))
test1 = np.reshape(mean_lat, (M,N))
test0 = np.reshape(mean_lon, (M,N))

vmin = np.nanpercentile(variable, 5)
vmax = np.nanpercentile(variable, 95)
cbar1 = ax.scatter(test0, test1, s = 1800, c = test2, cmap = 'magma', transform = crs, marker = 's', vmin=vmin, vmax=vmax)
cbaxes = fig.add_axes([0.92, 0.20, 0.03, 0.6]) 
cbar_loc = plt.colorbar(cbar1, cax = cbaxes)  
cbar_loc.set_label('Obukhov Length', fontsize=22, labelpad=10)
yg.scale_bar(ax, 100)

plt.show()



