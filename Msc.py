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

#%% calculate angle for alongwind rotation through PSD analysis

# load image and create Hamming window
image = Sigma0_VV
hamming_window = msc.HammingWindow(image)

# Filter image by dividing by longwave components and then apply Hamming window
image_longwave = msc.longWaveFilter(image, samplerate[1])
image_filtered = (image-image_longwave)/image_longwave*hamming_window #(image-image_longwave)/image_longwave*hamming_window
msc.plottings(image_filtered, samplerate[1], title = 'Longwave filtered and Hamming windowed input image')

# calculate 2D PSD for resulting image and apply Gaussian filter to Power Spectrum for smoothing
psd2D = msc.twoDPS(image_filtered, samplerate[1], True)
psd2D_filtered = ndimage.gaussian_filter(psd2D,3)
msc.plottings(10*np.log10(psd2D_filtered), samplerate[1], minP = 95, maxP = 99.9, unit = 'frequency', title = 'Gaussian filtered 2D spectrum, [dB]', cmap = 'viridis')
# msc.circ()

angle_spectra, data_fitted, _ = msc.peakShortwave(psd2D_filtered, samplerate[1], True)
# _, angle_wind = msc.GaussFit(psd2D_filtered, 0.005, samplerate[1], True)


#%% calculate wind origine and angle for best spectra

#############################
#### OPTIONAL ADD 90 DEG ####
#############################
# angle_spectra -=90; 
# angle_wind -=90

wind_origin_spectra = (azimuthAngle - angle_spectra) - 180
if wind_origin_spectra <= 0:
    wind_origin_spectra+= 360

# resolve ambiguity from OCN product
try:
    if (wind_origin_spectra - (direction_ocn + direction_ecmwf)/2) > 90:
        wind_origin_spectra+= 180
        if wind_origin_spectra >= 360:
            wind_origin_spectra-= 360
    print('Wind origin (spectra) w.r.t. North: %1.1i degrees' %wind_origin_spectra)
except:
    print('Wind origin (spectra) w.r.t. North: %1.1i +- 180 degrees' %wind_origin_spectra)
    pass    


#%% used derived angle to rotate, clip covert and calculate along wind 1D PSD and wind speed variance

# calculate clockwise rotation to allign image azimuth direction with wind direction
rotation = abs(wind_origin_spectra - rangeLookAngle) % 180 * -1*  np.sign(wind_origin_spectra - rangeLookAngle) 
# rotation += 90
# rotation = 0

# apply on sinc filtered image or not
image = Sigma0_VV    #   Sigma0_VV     Sigma0_VV_filt

# rotate image such that East West is along wind
image_rot =  msc.rotateAndClip(image, rotation, samplerate[1], True)
incident_angle_rotated = msc.rotateAndClip(incident, rotation, samplerate[1])

# submit into CMOD for windspeed 
windspeed = msc.applyCMOD(image_rot, wind_origin_spectra - rangeLookAngle, incident_angle_rotated, 15, samplerate[1], False, True)
# windspeed = msc.applyCMOD_IFR2(image_rot, wind_origin - rangeLookAngle, incident_angle_rotated, 15, samplerate[1], True)

# surface_stress0, friction_velocity0, z_00, Cdn0 = msc.loop1(windspeed)
# surface_stress, friction_velocity, z_0, Cdn, windfield, Charnock = msc.loop1_charnocks(windspeed, Cdn0)


variance_multiplier = 1.0 # !!!
windspeed = (windspeed - np.mean(windspeed)) * variance_multiplier + np.mean(windspeed)
    
#Calculate along wind spectrum of boundary layer
along_wind_psd = msc.psd1D(windspeed, samplerate[1], plotting = True, windowed = True, scaled = False, normalised = True)
variance_psd = msc.psd1D(windspeed, samplerate[1], plotting = True, windowed = True, scaled = False, normalised = False)

# determine spectral peak to find MABL depth 
Zi, powerlaw, smoothed_spectrum, peak_idx, x_axis, index2range, PSD = msc.sikora1997(windspeed, 
            samplerate[1], window_length = 5, smoothing_fact = 1.5, windowed = True, plotting = True)

print('Zi: %1.1i metres' %Zi)


#%% wind statistics

windspeed_mean = np.mean(windspeed)
windspeed_mean_u = np.mean(np.mean(windspeed, axis = 1))
windspeed_mean_v = np.mean(np.mean(windspeed, axis = 0))

windspeed_std = np.std(windspeed)
windspeed_std_u = np.mean(np.std(windspeed, axis = 1))
windspeed_std_v = np.mean(np.std(windspeed, axis = 0))

spectrum_variance = np.sqrt(np.sum(variance_psd[1:]))
print(r'Variance derived from spectrum (excluding DC): %1.2f m**2/s**2' %spectrum_variance)
print(r'Mean windspeed: %1.2f m/s' %windspeed_mean)


#%% Apply Young's calculation variance approach loop 1

karman = 0.40                           #
Charnock = 0.011                        #

gradients = np.gradient(windspeed)
gradient  = np.sqrt(gradients[0]**2 + gradients[1]**2)
normalised_gradient = (gradient-np.nanmin(gradient)) / (np.nanmax(gradient)-np.nanmin(gradient))
Charnock1 = normalised_gradient * (0.040-0.011) + 0.011

g = 9.8                                 # m/s**2
z = 10                                  # m
pi = 3.1415926535
rho_air = 1.2                           # kg/m**3, 1.2 for 20 degrees, 1.25 for 10 degreess                           
T_v = 298  
T = 20
nu = 1.326 * 10**(-5) *(1 + (6.542 * 10**(-3))* T + (8.301 * 10**(-6)) * T**2 - (4.840 * 10**(-9)) * T**3) # m**2/s

iterations = 10
A_friction_velocity = np.ones(iterations)    # m/s
A_surface_stress = np.ones(iterations)       # kg/ m / s**2  [Pa]
A_Cdn = np.ones(iterations)
A_z_0 = np.ones(iterations)

for i in range(iterations):
    if i > 0:
        A_friction_velocity[i] = np.sqrt(A_Cdn[i-1] * windspeed_mean**2)
        A_surface_stress[i] = rho_air * A_friction_velocity[i]**2
        A_z_0[i] = (Charnock * A_friction_velocity[i]**2) / g + 0.11 * nu / A_friction_velocity[i]
        A_Cdn[i] = (karman / np.log( z / A_z_0[i]) )**2
        
surface_stress = rho_air * A_Cdn[-1] * windspeed**2


#%% Apply Young's calculation variance approach loop 2A


b = 0.73
c = 0.253
B_friction_velocity = np.ones(iterations) * np.sqrt(np.mean(surface_stress) / rho_air)
B_z_0 = (Charnock * B_friction_velocity**2) / g + 0.11 * nu / B_friction_velocity
# B_friction_velocity = A_friction_velocity[-1]
# B_z_0 = A_z_0[-1]

B_Cd = np.ones(iterations)
B_Psi_m = np.ones(iterations)
B_x = np.ones(iterations)
B_L = np.array(np.ones(iterations))*np.inf*-1
B_windfield = []
B_S = np.ones(iterations) * np.mean(windspeed)
B_sigma_s = np.ones(iterations)
B_sigma_u = np.ones(iterations)  #* 0.45   #  0.226    1.13
B_sigma_u2 = np.ones(iterations)
B_B = np.ones(iterations)


for i in range(iterations):
    if i > 0:
        if np.isnan(Zi):
            print('Manually set Zi')
            
        B_x[i] = (1 + 16 * abs(z / B_L[i-1]))**0.25 # Young et al 2000 and Paulson 1970
        # B_x[i] = (1 + 16 * abs(z / B_L[i-1]))**0.333     # Young and kristensen 1992
        
        # # Young et al 2000 typo corrected
        B_Psi_m[i] = np.log(((1 + B_x[i]**2) / 2)**2) - 2 * np.arctan(B_x[i]) + pi / 2 
        
        # #Paulson 1970 and Sukanta lecture PBL3
        # B_Psi_m[i] = np.log( ((1+B_x[i]**2) / 2) * ((1 + B_x[i]) / 2)**2) - 2 * np.arctan(B_x[i]) + pi / 2 
        
        # # Young and kristensen 1992
        # B_Psi_m[i] = 1.5 * np.log((1 + B_x[i] + B_x[i]**2) / 3) - np.sqrt(3) * np.arctan(1/np.sqrt(3) *(B_x[i]-1) /(B_x[i]+1)) 
        
        # recompute windfield
        B_Cd[i] = (karman / (np.log(z / B_z_0[-1]) - B_Psi_m[i]) )**2
        windfield = np.sqrt(surface_stress / (B_Cd[i] * rho_air))
        B_windfield.append(windfield)
        B_S[i] = np.mean(windfield)

        B_sigma_s[i] = np.std(windfield)
        B_sigma_u2[i] = np.mean(np.std(windfield, axis = 1))
        B_sigma_u[i] =  B_sigma_s[i]
        
        # print(B_sigma_s[i] / B_S[i] <= 0.35)
        # print(B_sigma_u / B_friction_velocity)

        
        # # Panofski et al 1977 eq 5
        # B_L[i] = - Zi / (((B_sigma_u[i] / B_friction_velocity[i])**2 - 4) / 0.6)**(3/2) 
        
        # # Panofski et al 1977 eq 6
        # B_L[i] = - Zi / (2 * ((B_sigma_u[i] / B_friction_velocity[i])**3 - 12 ))  
        
        # # Wilson 2008
        B_L[i] = - Zi*b**(3/2) / ((((B_sigma_u[i] / A_friction_velocity[-1])**2 / (1 - (z/Zi)**c)) - 4))**(3/2) 
        
        # # Panofski et al 1977 and Caughey and Palmer (1979)
        # B_L[i] = - Zi / ((B_sigma_u[i] / A_friction_velocity[-1] / 0.6)**3 * karman)  
        # 
        B_B[i] =  - (B_friction_velocity[i]**3 * T_v) / (B_L[i] * karman * g)

#richardson number R_f
R_f = z / B_L[-1] / (1 - 15 * z / B_L[-1])**(-0.25) # Zechetto et al 1998
zi = ((B_sigma_u[-1] / B_friction_velocity)**3 -12) * B_L[-1] / (-0.5) # Zechetto et al 1998

# stability correction parameters
stab_cor1 = 1 - (B_Psi_m[-1]*np.sqrt(B_Cd[1])/karman)
stab_cor2 = np.sqrt(A_Cdn[-1]/B_Cd[-1])

#Panofski et al 
test_w_star = B_friction_velocity*(-Zi/(karman*B_L))**(1/3)
print('L loop B: %1.3f m' %B_L[-1])  
print('w* loop B: %1.3f m/s' %test_w_star[-1])
print('sigma_u loop B: %1.3f m/s' %B_sigma_u[-1])



#%%  Apply Young's calculation inertial subrange approach loop 2B

x_axis = 2*pi/(2*pi*(1/((1/np.arange(1,np.shape(along_wind_psd)[0]+1))*(2*samplerate[1]*np.shape(along_wind_psd)[0]))))

idx_kolmogorov = np.where((x_axis>650) & (x_axis<1000))
iterations = 10
kolmogorov = 0.5
dissip_rate = 2.6
T_v = 298               # Kelvin

C_w_star = np.ones(iterations)
C_B = np.ones(iterations)
C_L = np.ones(iterations)
C_x = np.ones(iterations)
C_Psi_m = np.ones(iterations)
C_corr_fact = np.ones(iterations)

for i in range(iterations):
    if i > 0:
        Lambda = x_axis[idx_kolmogorov]
        k = 2*pi*1/Lambda
        S = along_wind_psd[idx_kolmogorov] * C_corr_fact[i-1]**2
        n = 1/Lambda
        fi = n * Zi / (windspeed_mean * C_corr_fact[i-1])
        
        # Kaimal et al 1976 eq 5
            # Difference between 0.20 and 0.15, isotropy Kaimal et al 1976
        pre_w_star = np.sqrt((2 * pi)**(2/3) * fi**(2/3) * n * S / (4/3 * kolmogorov * dissip_rate**(2/3)))
        # pre_w_star = np.sqrt((2 * pi)**(2/3) * fi**(2/3) * n * S / (kolmogorov * dissip_rate**(2/3)))
        
        C_w_star[i] =  np.mean(pre_w_star)
        
        C_B[i] =  (C_w_star[i]**3 * T_v) / (g * Zi)

        C_L[i] = - (B_friction_velocity[-1]**3 * T_v) / (C_B[i] * karman * g)
        
        # young et al 2000
        C_x[i] = (1 + 16 * abs(z / C_L[i]))**0.25
        C_Psi_m[i] = np.log(((1 + C_x[i]**2) / 2)**2) - 2 * np.arctan(C_x[i]) + pi / 2 
        
        C_corr_fact[i] = 1 - (C_Psi_m[i] * np.sqrt(A_Cdn[-1])) / karman
 
C_sigma_u = B_friction_velocity * np.sqrt(4 + 0.6 * (-Zi / C_L[-1])**(2/3))
zi = ((C_sigma_u[-1] / B_friction_velocity)**3 -12) * C_L[-1] / (-0.5)      

print('L loop C: %1.3f m' %C_L[-1])  
print('w* loop C: %1.3f m/s' %C_w_star[-1])  
print('sigma_u loop C: %1.3f m/s' %C_sigma_u[-1])


#%% Loop 2C

# used for weighting w* values in inertial subrange
def weighted_avg_and_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))
    
form = 'rolls'  
pi = 3.1415926535
z = 10                                  # measurements height, 10 metres for CMOD5.N 
karman = 0.40                           # Karman constant
iterations = 10
kolmogorov = 0.5
U_mean = np.nanmean(windspeed)
friction_velocity = B_friction_velocity[-1]
Cdn = A_Cdn[-1]
PSD = along_wind_psd

#calculate x_axis of spatial wavenlengths
x_axis =    (1 / np.arange(1, np.shape(PSD)[0] + 1)) * (2 * samplerate[1] * np.shape(PSD)[0])   # wavelengths 
    
inertial_max = Zi * 1.5                          # longest wavelength to consider for inertial subrange (Zi * 1.5)
inertial_min = 650 #x_axis[idx_ll]      # shortest wavelength to consider for inertial subrange
    
# select inertial subrange to be between inertial_min and inertial_max metres
idx_kolmogorov = np.where((x_axis>inertial_min) & (x_axis < inertial_max))   
    
Lambda = x_axis[idx_kolmogorov]
S = PSD[idx_kolmogorov]
n = 1/Lambda
    
# create arrays to store loop results
D_etha = np.ones(iterations)
D_etha_std = np.ones(iterations)
D_L = np.ones(iterations)*-1
D_x = np.ones(iterations)
D_x2 = np.ones(iterations)
D_Psi_m = np.ones(iterations)
D_Psi_m2 = np.ones(iterations)
D_corr_fact = np.ones(iterations)
    
for i in range(iterations):
    if i > 0:
            
        # f = n * Zi / U_mean * D_corr_fact[i-1]
        
        if form != 'cells':
            pre_etha = (2 * pi / (U_mean * D_corr_fact[i-1])) * ( n**(5/3) * (S * D_corr_fact[i-1]**2) / kolmogorov / (4/3) )**(3/2)
        else:
            pre_etha = (2 * pi / (U_mean * D_corr_fact[i-1])) * ( n**(5/3) * (S * D_corr_fact[i-1]**2) / kolmogorov)**(3/2)
         
        weights = x_axis[idx_kolmogorov]/np.min(x_axis[idx_kolmogorov])
        D_etha[i] = weighted_avg_and_std(pre_etha, weights)[0]
        # D_etha[i] = 0.029*(U_mean*D_corr_fact[i-1])**(-3.24)
        D_etha_std[i] =  weighted_avg_and_std(pre_etha, weights)[1]
            
        Y = np.arange(-0.00001,-50,-0.00005)
        D_x = (1 + 16 * abs(Y))**(1/4)
        D_Psi_m = np.log(((1 + D_x**2) / 2)**2) - 2 * np.arctan(D_x) + pi / 2
        
        lhs = D_Psi_m - (Y)
        
        rhs = D_etha[i] * karman * z / friction_velocity**3 

        # find index with minimum difference between approaches
        idx_same = (np.abs(lhs - rhs)).argmin()
        
        D_L[i] = z / Y[idx_same]
        
        D_x2[i] = (1 + 16 * abs(z / D_L[i]))**(1/4)
        D_Psi_m2[i] = np.log(((1 + D_x2[i]**2) / 2)**2) - 2 * np.arctan(D_x2[i]) + pi / 2 
        
        # stability correction factor from young et al 2000
        D_corr_fact[i] = 1 - (D_Psi_m2[i] * np.sqrt(Cdn)) / karman

# calculate final outputs to return at the end of function
sigma_u = friction_velocity * np.sqrt(4 + 0.6 * (-Zi / D_L[-1])**(2/3)) 

D_etha = D_etha[-1]
D_etha_std = D_etha_std[-1]
corr_fact = D_corr_fact[-1]
    
print('L loop C: %1.3f m' %D_L[-1])  
print('sigma_u loop C: %1.3f m/s' %sigma_u)

idx_ll = None 
sigma_u2, L2, D_etha, D_etha_std, corr_fact2 = msc.loop2C(windspeed, friction_velocity, Zi, Cdn, PSD, samplerate[1], inertial_max, idx_ll, form = 'form')

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



