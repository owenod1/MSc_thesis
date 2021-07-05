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

#%% things to improve

"""
POINTS TO IMPROVE
1. Longwave filtering of image for angle determination, change to MEDIAN instead of MEAN
4. Improve 2d detrending ??
10. check Zi peak estimation
11. use windspeed field or mean of windspeed field to compute u_star in Loop 1A
        not the same because distribution is not normal 
12. value for dissipation rate (use 2 or so instead of 0.6 in loop 2B)

13. SET RIGHT VALUE FOR INERTIAL SUBRANGE LIMITS !!!

"""

#%% load SAR data

# barbedos_southeast3_ML300      barbedos_east1_ML300       
# young_ML150       barbedos_east5_ML300  
# barbedos_jan15b_ML300  barbedos_jan25a_ML300  barbedos_jan27b1_ML300   barbedos_jan27b2_ML300 
# barbedos_feb1b_ML300  barbedos_feb8b_ML300  barbedos_feb20b1_ML300 barbedos_feb20b2_ML300 
# delaware_ML300  alaska3_ML300  carolina_ML300  surface_wind
path = 'carolina_ML300.nc'

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


#%% apply low pass filter to image if swell is significant

# Sigma0_VV = msc.lowpass(Sigma0_VV, 51, samplerate[1]*3, samplerate[1], cmap = 'Greys_r', plotting = True)
Sigma0_VV_original = msc.lowpass(Sigma0_VV_original, 51, samplerate[1]*12, samplerate[1], cmap = 'Greys_r', plotting = True)

#%% tiled calculations 

# theta = 77 and zi = ... and threshold = 0.025 for 15b
# theta = 80 and zi = 500 and threshold = 0.020 for 27b1
# theta = 95 and zi = 500 and threshold = 0.015 for 27b2
# theta = 83 and zi = 650 and threshold = 0.020 for 01b
# theta = 77 and zi = 850 and threshold = 0.025 for 08b
# theta = 59 and zi = 800 and threshold = 0.030 for 20b2

True_wind = None #direction_ecmwf
Zi = None # 1100
size = int(83*1)    # 83
w_star_threshold = 0.15 #
dissip_rate =  1.8
slope_multiplier = 1.6
form = 'cells'

wind_origin, mean_lon, mean_lat, u_star, z_0, sigma_u1, sigma_u2, L1, L2, L3, w_star2, w_star2_std, Zi_estimated,\
    corr_fact1, corr_fact2, windspeed, tiles_lon, tiles_lat, idxKeep_tiles, hold_prediction, hold_form, epsilon,\
        H, dissip_rate_loop2B \
    = msc.tiledWind(Sigma0_VV_original, incident_original, longitude, latitude, iterations = 15, size = size, \
                    dissip_rate = dissip_rate, slope_multiplier = slope_multiplier, w_star_threshold = w_star_threshold, \
                        Zi_input = Zi, True_wind = True_wind, form = form, samplerate = samplerate[1], plotting = False)

# wind_origin, mean_lon, mean_lat, u_star, z_0, sigma_u1, sigma_u2, L1, L2, L3, w_star2, w_star2_std, Zi_estimated,\
#     corr_fact1, corr_fact2, windspeed, tiles_lon, tiles_lat, idxKeep_tiles, hold_prediction, hold_form, epsilon,\
#         H, dissip_rate_loop2B \
#     = msc.tiledWind(Sigma0_VV, incident, lon, lat, iterations = 15, size = size, \
#                     dissip_rate = dissip_rate, slope_multiplier = slope_multiplier, w_star_threshold = w_star_threshold, \
#                         Zi_input = Zi, True_wind = True_wind, form = 'cells', samplerate = samplerate[1], plotting = False)


# sets errorneous wind direction of 0 to nan
epsilon2 =  sigma_u2 / sigma_u2 * epsilon
wind_origin2 = sigma_u2 / sigma_u2 * wind_origin
hold_prediction = sigma_u2 / sigma_u2 * hold_prediction  
hold_form = sigma_u2 / sigma_u2 * hold_form

figFinal, pltsFinal = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), sharex = False, sharey = False)

bins = list(np.arange(0, 361, 15))
im1 = pltsFinal[0,0].hist(wind_origin2.flatten(), bins=bins, density=False,  alpha = 0.6, color= 'r', ls='dotted')
pltsFinal[0,0].set_title('Wind direction  [deg]', fontsize = 15)
pltsFinal[0,0].set_ylabel('Occurence [-]', fontsize = 12)

bins = list(np.arange(150, 2050, 150))
im2 = pltsFinal[0,1].hist(Zi_estimated.flatten(), bins=bins, density=False,  alpha = 0.6, color= 'g', ls='dotted')
pltsFinal[0,1].set_title('$Z_i$  [m]', fontsize = 15)
pltsFinal[0,1].set_ylabel('Occurence [-]', fontsize = 12)

bins = list(np.arange(0.1, 1.5, 0.02))
im2 = pltsFinal[0,2].hist(sigma_u2.flatten(), bins=bins, density=False,  alpha = 0.6, color= 'b', ls='dotted')
pltsFinal[0,2].set_title(r'$\sigma_u$  [m/s]', fontsize = 15)
pltsFinal[0,2].set_ylabel('Occurence [-]', fontsize = 12)

bins = np.logspace(np.log10(1),np.log10(10000), 50)
im2 = pltsFinal[1,0].hist(-L2[np.isnan(L2)==False].flatten(), bins = bins, density=False,  alpha = 0.6, color= 'k', ls='dotted')
pltsFinal[1,0].set_title(r'$L$  [m]', fontsize = 15)
pltsFinal[1,0].semilogx()
pltsFinal[1,0].set_ylabel('Occurence [-]', fontsize = 12)

bins = list(np.arange(0, 1.0, 0.02))
im2 = pltsFinal[1,1].hist(u_star.flatten(), bins=bins, density=False,  alpha = 0.8, color= 'C04', ls='dotted')
pltsFinal[1,1].set_title(r'$u_*$  [m/s]', fontsize = 15)
pltsFinal[1,1].set_ylabel('Occurence [-]', fontsize = 12)

bins = list(np.arange(0.5, 15.5, 0.5))
im2 = pltsFinal[1,2].hist(windspeed.flatten(), bins=bins, density=False,  alpha = 0.8, color= 'C08', ls='dotted')
pltsFinal[1,2].set_title(r'$\overline{U}$  [m/s]', fontsize = 15)
pltsFinal[1,2].set_ylabel('Occurence [-]', fontsize = 12)
figFinal.subplots_adjust(hspace = 0.4, wspace =0.3)


plt.figure()
bins = list(np.arange(0, 0.3, 0.001))
plt.hist(w_star2_std.flatten(), bins=bins, density=False,  alpha = 0.6, color= 'k', ls='dotted')


plt.figure(figsize=(8,6))
plt.semilogy(-L1, '-o', c = 'r', linewidth=2, label = r'Loop 2A')
plt.semilogy(-L2, '-o', c = 'g', linewidth=2, label = r'Loop 2B ($\psi= %1.1f$' %dissip_rate + ')')
plt.semilogy(-L3, '-o', c = 'b', linewidth=2, label = r'Loop 2C')
plt.title(r'Obukhov results, slope multiplier: %1.1f' %slope_multiplier +'', fontsize = 15)
plt.legend(loc='best', fontsize = 12)
plt.ylabel(r'$-L/m$', fontsize = 15)
plt.xlabel(r'Tile', fontsize = 15)


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


z = 10
kappa = 0.4
corr_factor = epsilon * z * kappa / u_star**3
np.nanmean(corr_factor)

#%% load OCN data and ECMWF data

# ocn_barbedos_east1   ocn_barbedos_southeast3 
# ocn_barbedos_jan15b
# ocn_barbedos_jan25a  ocn_barbedos_jan25b
# ocn_barbedos_jan27b1   ocn_barbedos_jan27b2   ocn_barbedos_feb1b
# ocn_barbedos_feb8b   ocn_barbedos_feb20b1   ocn_barbedos_feb20b2
# ocn_delaware  ocn_alaska3  ocn_carolina

path = 'ocn_carolina.nc' 
netcdf_file = nc.Dataset(path, 'r', format='NETCDF4')
# print(netcdf_file)

wind_ocn = netcdf_file['vv_001_owiWindSpeed'][:]
direction_pre_ovn = np.array(netcdf_file['vv_001_owiWindDirection'][:])
direction_ocn = np.array(netcdf_file['vv_001_owiWindDirection'][:])[:,direction_pre_ovn[0,:]!=-999]
direction_ocn_median, direction_MAD_ocn  = np.median(direction_ocn), msc.MAD(direction_ocn)
longitude_wind = np.array(netcdf_file['lon'][:])
latitude_wind = np.array(netcdf_file['lat'][:])
samplerate_wind = msc.sampleRate(longitude_wind,latitude_wind)
windspeed2 = np.array(wind_ocn)
windspeed2 = windspeed2[:,windspeed2[0,:]!=-999]
msc.plottings(windspeed2, samplerate_wind[1], title = 'S1 OCN windspeed data [m/s]', cmap = 'jet')
msc.plottings(direction_ocn, samplerate_wind[1], title = 'S1 OCN wind direction [deg]', cmap = 'Reds')

# ecmwf_barbedos_east1  ecmwf_barbedos_east5     ecmwf_barbedos_southeast3
# ecmwf_barbedos_jan15b  ecmwf_barbedos_jan25a    ecmwf_barbedos_jan25b
# ecmwf_barbedos_jan27b1  ecmwf_barbedos_jan27b2   ecmwf_barbedos_feb1b
# ecmwf_barbedos_feb8b  ecmwf_barbedos_feb20b1   ecmwf_barbedos_feb20b2
# ecmwf_delaware   ecmwf_alaska3    ecmwf_carolina

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
msc.plottings(windspeed3, samplerate_wind[1], title = 'OCN windspeed data [m/s]', cmap = 'jet')
msc.plottings(np.array(netcdf_file['vv_001_owiEcmwfWindDirection'][:]), samplerate_wind[1], title = 'ECMWF wind direction [deg]', cmap = 'Reds')


print(r'OCN Wind origin w.r.t. North: %1.1f degrees' %direction_ocn_median + '+- %1.1f degrees' %direction_MAD_ocn)
print(r'ECMWF Wind origin w.r.t. North: %1.1f degrees' %direction_ecmwf_median + '+- %1.1f degrees' %direction_MAD_ecmwf)
print(r'OCN Wind speed: %1.1f m/s' %np.mean(windspeed2) + '+- %1.1f m/s' %np.std(windspeed2))
print(r'ECMWF Wind speed: %1.1f m/s' %np.mean(windspeed3) + '+- %1.1f m/s' %np.std(windspeed3))


#%% # clip and visualise part of radar image

n = 1/1
# beginx = 150; endx = 400; beginy = 175; endy = 425  # barbedos north3
# beginx = 250; endx = 300; beginy = 275; endy = 325  # alaska 2
# beginx = 50; endx = 150; beginy = 450; endy = 550  # alaska 1
# beginx = 150; endx = 400; beginy = 175; endy = 425  # young 10  (n = 1/12.5)
# beginx = 150; endx = 400; beginy = 175; endy = 425  # young 150
# beginx = 150; endx = 300; beginy = 175; endy = 325  # young
# beginx = 250; endx = 450; beginy = 0; endy = 200  # southeast6 ML300 
# beginx = 50; endx = 300; beginy = 175; endy = 425  # southeast5 ML300 
# beginx = 1500; endx = 4000; beginy = 1750; endy = 4250  # young
# beginx = 2000; endx = 3000; beginy = 1500; endy =2500 # barbedos_southeast1
# beginx = 50; endx = 250; beginy = 300; endy = 500 # barbedos_feb20b2
# beginx = 250; endx = 500; beginy = 100; endy = 350 # barbedos_feb20b2
beginx = 250; endx = 450; beginy = 250; endy = 450 # barbedos_feb20b2 overpass
# beginx = 250; endx = 350; beginy = 250; endy = 350 # barbedos_feb20b2 overpass subset
# beginx = 100; endx = 300; beginy = 250; endy = 450 # barbedos feb1b
# beginx = 450; endx = 650; beginy = 0; endy = 200 # barbedos feb1b overpass
# beginx = 550; endx = 750; beginy = 200; endy = 400 # barbedos_feb8b_ML300
# beginx = 300; endx = 500; beginy = 150; endy = 350 # barbedos_jan27b1_ML300
# beginx = 50; endx = 250; beginy = 50; endy = 250 # barbedos_jan27b2_ML300
# beginx = 0; endx = 200; beginy = 300; endy = 500 # barbedos_jan15b_ML300
# beginx = 550; endx = 750; beginy =50; endy = 250 # barbedos_jan25a_ML300
# beginx = 600; endx = 800; beginy = 300; endy = 500 # barbedos_jan25b_ML300
# 
# beginx = 50; endx = 120; beginy = 50; endy = 120
 
 
#[200:500,200:500]#[1000//3:3000//3,1000//3:3000//3]#[200:500,100:600]#[1000:3000,1000:3000]#[0000:500,0000:500]#[0000:1000,0000:1000]
Sigma0_VV = Sigma0_VV_original[int(beginy//n):int(endy//n),int(beginx//n):int(endx//n)]         #[000//n:400//n,0000//n:400//n]
incident = incident_original[int(beginy//n):int(endy//n),int(beginx//n):int(endx//n)]           #[0000//n:400//n,0000//n:400//n]
lat = latitude[int(beginy//n):int(endy//n),int(beginx//n):int(endx//n)]         #[000//n:400//n,0000//n:400//n]
lon = longitude[int(beginy//n):int(endy//n),int(beginx//n):int(endx//n)]    
msc.plottings(Sigma0_VV, samplerate[1], title = 'Subset NRCS  $\sigma_0$')

#%% spatial smoothing or low pass filter

Sigma0_VV = msc.lowpass(Sigma0_VV, 101, samplerate[1]*12, samplerate[1], cmap = 'Greys_r', plotting = True)

#%% determination of validation sigma_u

Zi = np.array([500, 500, 650, 650, 850, 850, 800, 800])
L = np.array([-45, -29, -126, -87, -495, -219, -358, np.nan])
friction_velocity = np.array([0.13, 0.16, 0.22, 0.27, 0.35, 0.36, 0.30, np.nan])
sigma_u = friction_velocity * np.sqrt(4 + 0.6 * (-Zi / L)**(2/3))

Zi_test = 800
L_test = -403
u_test = 0.30
sigma_u_test = u_test * np.sqrt(4 + 0.6 * (-Zi_test / L_test)**(2/3))
print(sigma_u_test)

#%% plot wind direction on map

import cartopy
import cartopy.crs as ccrs

angles =np.where(wind_origin2>=180,wind_origin2-180,wind_origin2 )
 
# angles = wind_origin
mean_total_lon = np.mean(longitude)
mean_total_lat = np.mean(latitude)

plus = 1.5
extent = [mean_total_lon-plus, mean_total_lon+plus, mean_total_lat-plus, mean_total_lat+plus]
# extent2 = [, mean_total_lon+2, mean_total_lat-2, mean_total_lat+2]



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
# ax.set_global()
ax.gridlines()

msc.scale_bar(ax, 100)

ax.quiver(mean_lon, mean_lat, v, u, scale=14, transform = crs)
plt.show()

#%% plot other variables on map

import cartopy
import cartopy.crs as ccrs
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh


plus = 2.5
mean_total_lon = np.mean(mean_lon)
mean_total_lat = np.mean(mean_lat)
extent = [mean_total_lon-plus, mean_total_lon+plus, mean_total_lat-plus, mean_total_lat+plus]
crs = ccrs.RotatedPole(pole_longitude=180, pole_latitude=90)



fig, ax = plt.subplots(figsize=(15,12))

ax = plt.axes(projection=ccrs.Orthographic(mean_total_lon, mean_total_lat))
ax.add_feature(cartopy.feature.OCEAN, zorder=0)
ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black')
ax.set_extent(extent)
# ax.set_global()
# ax.gridlines()

# u_star, z_0, sigma_u1, sigma_u2, L1, L2, w_star2, w_star2_std, Zi_estimated, hold_prediction, 
# hold_form, windspeed, epsilon, H
im = Sigma0_VV_original_preclip
M, N = np.array(np.shape(im))//size
test2 = np.reshape(hold_form, (M,N))
test1 = np.reshape(mean_lat, (M,N))
test0 = np.reshape(mean_lon, (M,N))

# cbar1 = ax.pcolormesh(test0, test1, test2, transform = crs)
cbar1 = ax.scatter(test0, test1, s = 1000, c = test2, transform = crs, marker = 's')
plt.colorbar(cbar1,ax=ax)
# cbar1.set_clim(0,1)
plt.show()



