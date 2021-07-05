# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:35:19 2021

@author: Owen O'Driscoll
"""

# %reset -f

import numpy as np  
from matplotlib import pyplot as plt  
import os  
import math

os.chdir('C:/Python/AAA_GRS2/mscthesis/')


#%% plot data

def plottings(image, samplerate = 50, minP = 1, maxP = 99, unit = 'metres', title= 'title', cmap = 'Greys_r'):
    if unit == 'metres':
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        plt.title(title)
        plt.ylabel('Azimuth [Km]')
        plt.xlabel('Range [Km]')
        vmin= np.nanpercentile(image,minP)
        vmax= np.nanpercentile(image,maxP)
        xend = image.shape[1]*samplerate/1000  #divide by 1000 to go from metre to km
        yend = image.shape[0]*samplerate/1000  #divide by 1000 to go from metre to km
        cbar1 = ax.imshow(image, vmin = vmin, vmax = vmax, cmap=cmap, origin = 'lower', extent=[0,xend,0,yend])
        plt.colorbar(cbar1, fraction=0.031, pad=0.05)

    if unit == 'frequency':
        from matplotlib.ticker import MaxNLocator
        pi = 3.1415926535
        axis = 2*pi*(1/((1/np.arange(1,np.shape(image)[0]//2+1))*(2*samplerate*np.shape(image)[0]//2)))

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        ax.yaxis.set_major_locator(MaxNLocator(8))
        ax.xaxis.set_major_locator(MaxNLocator(8)) 
        plt.title(title)
        plt.ylabel(r'$k$ [$rad\ m^{-1}$]')
        plt.xlabel(r'$k$ [$rad\ m^{-1}$]')
        vmin=np.nanpercentile(image,minP)
        vmax= np.nanpercentile(image,maxP)
        cbar1 = ax.imshow(image, vmin = vmin, vmax = vmax, cmap=cmap, origin = 'lower', extent=[-axis[-1],axis[-1],-axis[-1],axis[-1]])
        plt.colorbar(cbar1,ax=ax)
        # plt.locator_params(axis='y', nbins=9)
        # plt.locator_params(axis='x', nbins=9)


#%% plot data with grid after tiled calculations

def gridded(longitude, latitude, image, tiles_lon, tiles_lat, idxKeep_tiles, mean_lat, mean_lon, size_plot):
    
    edges_lon = []
    edges_lat = []
    plt.figure(figsize = (10,8))
    
    #plot radar image
    plt.scatter(longitude, latitude, 0.5, c = image, cmap = 'Greys_r', \
            vmin = np.percentile(image, 1), vmax = np.percentile(image, 99))
    plt.ylabel('Latitude [deg]', fontsize = 12)
    plt.xlabel('Longitude [deg]', fontsize = 12)
    plt.title(r'Tile size: %1.1f km$^2$' %size_plot, fontsize = 15)    
    
    for i in range(len(idxKeep_tiles)):
        placeholder_lon = np.ones_like(tiles_lon[idxKeep_tiles[i]])*np.nan
        placeholder_lon[0,:] = tiles_lon[idxKeep_tiles[i]][0,:]
        placeholder_lon[-1:] = tiles_lon[idxKeep_tiles[i]][-1:]
        placeholder_lon[:,0] = tiles_lon[idxKeep_tiles[i]][:,0]
        placeholder_lon[:,-1] = tiles_lon[idxKeep_tiles[i]][:,-1]
        edges_lon.append(placeholder_lon)
 
        placeholder_lat = np.ones_like(tiles_lat[idxKeep_tiles[i]])*np.nan
        placeholder_lat[0,:] = tiles_lat[idxKeep_tiles[i]][0,:]
        placeholder_lat[-1:] = tiles_lat[idxKeep_tiles[i]][-1:]
        placeholder_lat[:,0] = tiles_lat[idxKeep_tiles[i]][:,0]
        placeholder_lat[:,-1] = tiles_lat[idxKeep_tiles[i]][:,-1]
        edges_lon.append(placeholder_lat)

        plt.text(x=mean_lon[i], y=mean_lat[i], s = i+1, color = 'r')
        plt.scatter(placeholder_lon, placeholder_lat,0.02, color = 'r')
        
    return edges_lon, edges_lat


#%% detrend

def trend(image, plotting = False):
    """
    https://stackoverflow.com/questions/33964913/equivalent-of-polyfit-for-a-2d-polynomial-in-python
    
    """
    x = np.linspace(0, len(image)-1, len(image))
    y = np.linspace(0, len(image)-1, len(image))
    X, Y = np.meshgrid(x, y, copy=False)
    Z = image

    X = X.flatten()
    Y = Y.flatten()

    A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
    B = Z.flatten()

    coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=None)


    def poly2Dreco(X, Y, c):
        return (c[0] + X*c[1] + Y*c[2] + X**2*c[3] + X**2*Y*c[4] + X**2*Y**2*c[5] + 
                Y**2*c[6] + X*Y**2*c[7] + X*Y*c[8])

    trend = poly2Dreco(X, Y, coeff).reshape((len(image),len(image)))
    
    if plotting == True:
        plottings(trend, title= 'linear trend image')
    return trend


#%% 2D power spectrum

def twoDPS(image, samplerate = 50, plotting = False):
    """
    Calculate 2D power spectrum of input NRCS data 
    """
    
    F1 = np.fft.fft2(np.array(image))
    # low spatial frequencies are in the center of the 2D fourier transformed image.
    F2 = np.fft.fftshift( F1 )
    # Calculate a 2D power spectrum
    psd2D = np.abs( F2)**2
    
    h  = psd2D.shape[0]
    w  = psd2D.shape[1]
    wc = w//2
    hc = h//2
    
    if plotting == True:
        vmin = np.percentile(10*np.log10(psd2D),5)
        vmax = np.percentile(10*np.log10(psd2D),99)
        pi = 3.1415926535
        axis = 2*pi*(1/((1/np.arange(1, h // 2+1))*(2*samplerate* h //2)))
        
        fig, (ax1) = plt.subplots(1, 1,figsize=(8,6))
        plt.title('Unfiltered two-dimensional power spectra [dB]')
        plt.ylabel(r'$k$ [$rad\ m^{-1}$]')
        plt.xlabel(r'$k$ [$rad\ m^{-1}$]')
        cbar1 = ax1.imshow(10*np.log10(psd2D), vmin=vmin, vmax=vmax , origin = 'lower', extent=[-axis[-1],axis[-1],-axis[-1],axis[-1]])
        plt.colorbar(cbar1,ax=ax1)
    return psd2D 


#%% plot circles

def circ(plotting = True):
    pi = 3.1415926535
    rad600 = 0.010471975511666667
    circx100 = np.cos(np.arange(0,2*pi,0.01))*rad600*6/1
    circy100 = np.sin(np.arange(0,2*pi,0.01))*rad600*6/1
    circx300 = np.cos(np.arange(0,2*pi,0.01))*rad600*6/3
    circy300 = np.sin(np.arange(0,2*pi,0.01))*rad600*6/3
    circx600 = np.cos(np.arange(0,2*pi,0.01))*rad600
    circy600 = np.sin(np.arange(0,2*pi,0.01))*rad600
    circx1200 = np.cos(np.arange(0,2*pi,0.01))*rad600/2
    circy1200 = np.sin(np.arange(0,2*pi,0.01))*rad600/2

    plt.plot(circy100,circx100, 'w--', linewidth= 2.5)
    plt.plot(circy300,circx300, 'w--', linewidth= 2.5)
    plt.plot(circy600,circx600, 'w--', linewidth= 2.5)
    plt.plot(circy1200,circx1200, 'w--', linewidth= 2.5)
    # plt.text(circx100[len(circx100)//8] - 0.02, circy100[len(circx100)//4] - 0.01, '100m', color = 'w', fontsize = 12, weight = 'bold')
    # plt.text(circx300[len(circx300)//8] + 0.0008, circy300[len(circx300)//4] + 0.0005, '300m', color = 'w', fontsize = 12, weight = 'bold')
    # plt.text(circx600[len(circx600)//8] + 0.012 , circy600[len(circx600)//8] + 0.005, '600m', color = 'w', fontsize = 12, weight = 'bold')
    # plt.text(circx1200[len(circx1200)//8]+ 0.016, circy1200[len(circx1200)//8], '1200m', color = 'w', fontsize = 12, weight = 'bold')

    plt.text(-0.039, 0.040, '100m', color = 'w', fontsize = 12, weight = 'bold')
    plt.text(-0.039, 0.01, '300m', color = 'w', fontsize = 12, weight = 'bold')
    plt.text(-0.039, 0.0, '600m', color = 'w', fontsize = 12, weight = 'bold')
    plt.text(-0.039, -0.01, '1200m', color = 'w', fontsize = 12, weight = 'bold')

#%% calculate Median of Absolute Deviations

def MAD(data):
    median = np.nanmedian(data)
    MAD = np.nanmedian(abs(data - median))
    return MAD


#%% compute sampling rate

def sampleRate(longitude, latitude):
    pi = 3.14159265358979

    origin = np.array([longitude[0,0],latitude[0,0]])*pi/180
    maxlat = np.array([longitude[0,-1],latitude[0,-1]])*pi/180
    maxlon = np.array([longitude[-1,0],latitude[-1,0]])*pi/180

    # 
    dlat = (maxlon[1]-origin[1])
    dlon = (maxlon[0]-origin[0])
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(origin[1]) \
             * np.cos(maxlon[1]) * np.sin(dlon/2) * np.sin(dlon/2)
    c = np.arcsin(np.sqrt(a))
    radius = 6371000 # (km)
    distance1 = radius * c * 2 
    samplerate1 = np.rint(distance1/np.shape(longitude)[0])

    #
    dlat = (maxlat[1]-origin[1])
    dlon = (maxlat[0]-origin[0])
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(origin[1]) \
             * np.cos(maxlat[1]) * np.sin(dlon/2) * np.sin(dlon/2)
             
    c = np.arcsin(np.sqrt(a))
    radius = 6371000 # (km)
    distance2 = radius * c * 2 
    samplerate2 = np.rint(distance2/np.shape(longitude)[1])
    samplerate= [samplerate1,samplerate2]
    return samplerate


#%% hamming window to apply on backscatter image

def HammingWindow(image, plotting = False):
    
    """
    Apply Hamming window on input NRCS data
    """
    #create 1D Hamming windows
    windowx = np.hamming(image.shape[1]).T
    windowy = np.hamming(image.shape[1])
    
    #meshgrid to combine both 1-D filters into 2D filter
    windowX, windowY = np.meshgrid(windowx, windowy)
    window = windowX*windowY
    
    #plot filter
    if plotting == True:
        fig = plt.figure(figsize=(10,5))
        plt.title('Hamming window filter')
        ax = fig.add_subplot(111)
        cbar1 = ax.imshow(window)
        plt.colorbar(cbar1,ax=ax)
    return window


#%% apply sinc kernel for low pass filtering

def lowpass(image, kernelSize, CutOffWavelength, samplerate, cmap = 'Grey', plotting = False):
    from scipy import signal
    
    fc = samplerate / CutOffWavelength   # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    
    N = kernelSize
    if not N % 2: N += 1  # Make sure that N is odd.

    # Compute sinc filter.
    h  = N; w  = N
    wc = w/2; hc = h/2
    Y, X = np.ogrid[-hc:hc, -wc:wc]

    # compute sinc function
    s = np.sinc(2 * fc * X) * np.sinc(2 * fc * Y)

    # Compute Hamming window.
    w = HammingWindow(s)

    # Multiply sinc filter by window.
    f = s * w
 
    # Normalize to get unity gain.
    kernel = f / np.sum(f)
    
    #apply kernel and calculate spectrum of filtered image
    image_filt = signal.convolve2d(image, kernel, boundary='symm', mode='same')
    image_filt_freq = twoDPS(image_filt, samplerate)
    
    #plot frequency domain response of filter
    freq_domain = abs(np.fft.fftshift(np.fft.fft2(kernel)))
    
    if plotting == True:
        plottings(image_filt, samplerate, unit = 'metres', title = 'low pass filtered image', cmap = cmap)
        plottings(image_filt_freq, samplerate, unit = 'frequency', title = 'frequency response low pass filtered image', cmap = 'viridis')
        plt.figure()
        plt.plot(freq_domain[N//2])
    return image_filt  


#%% Calculate long wave filter

# ##### change Gaussian to median filter!!! ##########

def longWaveFilter(image, samplerate, plotting = False):
    import cv2
    from scipy import ndimage
    """
    Calculate long wave components (>10km) of NRCS image.
    Use remove these components by dividing NRCS by output 
    Source: Wakkerman 1996
    """
    # calculate how many pixels a 10kmx10km filter should be
    kernelSize =  int(10000//samplerate)
    if kernelSize%2!= 1:
        kernelSize+=1
    
    # apply Gaussian blur kernel !!!!!! CHANGE TO MEDIAN FILTER !!!!!!!!
    image_longwave = cv2.GaussianBlur(image, (kernelSize , kernelSize), 0)
    
    # image_longwave = ndimage.median_filter(image, size=kernelSize)
    
    # filterPrep = np.int8(30*np.log10(Sigma0_VV_original))
    # image_longwavepre = ndimage.median_filter(filterPrep, size=50)
    # image_longwave = 10**(image_longwavepre/30)
    
    
    
    #plot long wave componenets
    if plotting == True:
        plottings(image_longwave)
    return image_longwave


#%% filter wavelengths shorter than 750 meter in 2D Powerspectrum for deriving wind direction

def filter2DPS(psd2D, samplerate, minimumWavelength, plotting = False):
    """
    Remove wavelengths shorter than specified minimum wavelength 
    """
    
    h  = psd2D.shape[0]
    w  = psd2D.shape[1]
    wc = w//2
    hc = h//2
    Y, X = np.ogrid[0:h, 0:w]
    r    =(1/np.hypot(X - wc, Y - hc).astype(np.int)*2*samplerate*(len(psd2D)//2))

    idxRelevant = np.where(r>minimumWavelength,psd2D,0)

    idx2showbeginY = np.argmax(idxRelevant[len(idxRelevant)//2,:]!=1)
    idx2showendY = len(idxRelevant[len(idxRelevant)//2,:]) - idx2showbeginY
    idx2showbeginX = np.argmax(idxRelevant[:,np.shape(idxRelevant)[1]//2]!=1)
    idx2showendX = len(idxRelevant[:,np.shape(idxRelevant)[1]//2]) - idx2showbeginX

    data2fit = 10*np.log10(psd2D[idx2showbeginX:idx2showendX,idx2showbeginY:idx2showendY])
    if plotting == True:
        vmax = np.percentile(10*np.log10(idxRelevant[idxRelevant!=1]),90)
        vmin = np.percentile(10*np.log10(idxRelevant[idxRelevant!=1]),10)

        fig = plt.figure(figsize=(10,5))
        plt.title('Wavelengths > 0.75km  \n longwave filtered two-dimensional power spectra')
        ax = fig.add_subplot(111)
        cbar1 = ax.imshow(10*np.log10(idxRelevant[idx2showbeginX:idx2showendX,idx2showbeginY:idx2showendY]), vmin = vmin, vmax =vmax, origin = 'lower')
        plt.colorbar(cbar1,ax=ax)
    return data2fit


#%% Calculate angle of short wave components 

def peakShortwave(psd2D, samplerate, plotting = False):
    from scipy import ndimage
    from matplotlib.patches import Arrow
    """
    Calculated the local peak in a 2D PSD 
    between spectral wavelengths of 300 and 1000 meter (approx lower bound of roll vortices)
    """
    pi = 3.1415926535
    h  = psd2D.shape[0]
    w  = psd2D.shape[1]
    wc = w//2
    hc = h//2
    Y, X = np.ogrid[0:h, 0:w]
    r    =(1/np.hypot(X - wc, Y - hc).astype(np.int)*2*samplerate*(len(psd2D)//2))
    
    # psd2D = psd2D_filtered
    psd2D_weighted = psd2D #10**(psd2D/10)/r
    psd2D_weighted_averaged = ndimage.gaussian_filter(psd2D_weighted,3)
    psd2D_weighted_averaged_clipped = np.where((r<800) | (r>5000), np.min(psd2D_weighted_averaged), psd2D_weighted_averaged)
    
    idx_max = np.unravel_index(np.argmax(np.where(Y>=hc,psd2D_weighted_averaged, -99)), psd2D_weighted_averaged.shape)

    if idx_max[1]< wc:
        O = abs(idx_max[0]-hc)
        A = abs(idx_max[1]-wc)
        angle = 90 -np.arctan(O/A)*180/pi
    else:
        O = abs(idx_max[0]-hc)
        A = abs(idx_max[1]-wc)
        angle2 = np.arctan(O/A)*180/pi
        angle =  -(90- angle2)


    if plotting == True:
        
        fig, (ax1) = plt.subplots(1, 1,figsize=(8,6))
        # circ()
        axis = 2*pi*(1/((1/np.arange(1,np.shape(psd2D)[0]//2+1))*(2*samplerate*np.shape(psd2D)[0]//2)))
        vmin = np.percentile(psd2D_weighted_averaged,95.5)
        vmax = np.percentile(psd2D_weighted_averaged,99.9)
        plt.ylabel('Wavenumber [rad / metre]')
        plt.xlabel('Wavenumber [rad / metre]')
        plt.title('Spectral peak in Gaussian filtered 2D psd [arbitrary]')
        cbar1 = ax1.imshow(psd2D_weighted_averaged, cmap = 'viridis', origin = 'lower', vmin = vmin , vmax = vmax, extent=[-axis[-1],axis[-1],-axis[-1],axis[-1]])
        
        plt.colorbar(cbar1,ax=ax1)
        
        idx_max_rad_0 = 2*pi/(1/(idx_max[0]-wc)*(samplerate*np.shape(psd2D)[0]))
        idx_max_rad_1 = 2*pi/(1/(idx_max[1]-hc)*(samplerate*np.shape(psd2D)[0]))
        

        dy, dx = (idx_max_rad_1, idx_max_rad_0) 
        patches = Arrow(0, 0, dy, dx, width=0.002, color='red')
        
        ax1.add_patch(patches)
        plt.scatter(dy, dx, s=30, c='red', marker='o')
        plt.show()
    return angle, psd2D_weighted_averaged_clipped, r


#%% rotate AND clip

def rotateAndClip(image, rotation, samplerate, plotting = False):
    from scipy import ndimage
    """
    Clip the image such that only parts of the original image is available

    Optimum clip retrieved from 
    https://math.stackexchange.com/questions/828878/calculate-dimensions-of-square-inside-a-rotated-square
    """
    
    pi = 3.14159265358979
    
    # rotate image
    image_rotated = ndimage.rotate(image, rotation)
    
    # determine remaining angle (90 degree rotations dont need clipping)
    # so determine remaining angle after subtracting integer multiple of 90
    mult = abs(rotation) // 90
    if mult > 0:
        rem_rotation = abs(rotation) % (90 * mult) * np.sign(rotation)
    else:
        rem_rotation = rotation
    
    # define shape of input image
    h  = image.shape[0]
    w  = image.shape[1]
    
    # formulae retrieved from attached link
    a = int(np.ceil(h/(np.cos(abs(rem_rotation)*pi/180)+np.sin(abs(rem_rotation)*pi/180))))
    b = int(np.ceil(a*np.cos(rem_rotation*pi/180)))
    c = int(np.ceil(np.sin(abs(rem_rotation)*pi/180)*b))
    d = int(np.ceil((h-b)*np.sin((90-abs(rem_rotation))*pi/180)))
    
    # add a couple -1's and +1's to indexes in order to remove empty data on the edges
    image_rotated_clipped = image_rotated[c+1:(c + a )-1, d+1:(d + a)-1]

    if plotting == True:
        plottings(image_rotated_clipped, samplerate, title = 'NRCS  $\sigma_0$, clipped rotated image')
    return image_rotated_clipped


#%% calculate range angle

def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
      
      
    https://gist.github.com/jeromer/2005586
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    # compass_bearing = initial_bearing

    return compass_bearing

    
#%% code from stackoverflow to add scale bar to cartopy maps

def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=5):
    import cartopy.crs as ccrs
    
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    
    retrieved from:
    https://stackoverflow.com/questions/32333870/how-can-i-show-a-km-ruler-on-a-cartopy-matplotlib-plot
    
    """
    
    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length: 
        length = (x1 - x0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)        
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length) 

    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    #Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=linewidth)
    #Plot the scalebar label
    ax.text(sbx, sby, str(length) + ' km', transform=tmc,
            horizontalalignment='center', verticalalignment='bottom', fontsize = 30)


#%% Apply CMOD5.n on rotated image

def applyCMOD(NRCS, phi, incidence, iterations, samplerate = 50, CMOD5 = False, plotting = False):
    import cmod5_test
    
    """
    code retrieved from 
    https://gitlab.tudelft.nl/drama/stereoid/-/blob/a8a31da38369a326a5172d76a51d73bba8bc2d58/stereoid/oceans/cmod5n.py
    
    phi = 0 if the image is rotated
    """
    
    # estimate windspeed using CMOD function
    windspeed = cmod5_test.cmod5n_inverse(NRCS, phi, incidence, CMOD5, iterations = iterations)

    # plot windspeed
    if plotting == True:
        if CMOD5 == True:
            plottings(windspeed, samplerate, title = 'CMOD5 windspeed [m/s]', cmap = 'jet')
        else:
            plottings(windspeed, samplerate, title = 'CMOD5.n windspeed [m/s]', cmap = 'jet')
    return windspeed


#%% Apply CMOD_IFR2 on rotated image

def applyCMOD_IFR2(NRCS, phi, incidence, iterations, samplerate = 50, plotting = False):
    import CMOD_IFR2
    
    """
    fucntion to call CMOD IFR 2
    
    code skeleton retrieved from 
    https://gitlab.tudelft.nl/drama/stereoid/-/blob/a8a31da38369a326a5172d76a51d73bba8bc2d58/stereoid/oceans/cmod5n.py

    """
    # estimate windspeed using CMOD function
    windspeed = CMOD_IFR2.cmodIFR2_inverse(NRCS, phi, incidence, iterations = iterations)

    # plot windspeed
    if plotting == True:
        plottings(windspeed, samplerate, title = 'CMOD IFR2 windspeed [m/s]', cmap = 'jet')
    return windspeed


#%% calculate 1D PSD

def psd1D(image, samplerate, plotting= False, windowed = False, scaled = False, normalised = False):
    
    """
    Spectra calculated following Stull (1988)
    
    angle input 
    
    other source:
    https://www.ap.com/blog/fft-spectrum-and-spectral-densities-same-data-different-scaling/
    https://www.sjsu.edu/people/burford.furman/docs/me120/FFT_tutorial_NI.pdf 
    
    input:
        image: 2D wind speed image rotated that greatest avraince direction is either orientated horizontal or vertical
        samplerate: image samplerate
        angle: not used
        plotting: boolean, whether to plot output PSD
        windowed: boolean, whether to apply a 1D filter to per row basis prior to calculation of PSD, useful in most cases
        scaled: boolean, whether to multiply PSD with frequency axis, only used for peak determination following sikora 1997
        normalised: boolean, whether to normalise such that output is energy spectrum (False) or power spectrum PSD (True)
                    sum of energy spectrum yields variance
    
    output:
        1D spectrum, either energy spectrum or power spectrum depending on normalisation

    """
    
    pi = 3.14159265358979
    
    if windowed == True:
        # apply window on per row basis, Hanning window for small amplitude (microscale peak) component far off a large amplitude component (near DC)
        window_0 = np.hanning(image.shape[1])  # 
        
        # turn into 2D window with orientation following rows
        base = np.ones(image.shape[0])
        window1 = np.outer(base, window_0)
        
        # normalise window
        window = window1 / np.mean(window1)
    else:
        base = np.ones(image.shape[0])
        window = np.outer(base, base)
    
    N = np.shape(image)[1]                          # number of points in original FFT
    fs = 1 / samplerate                             # samplerate
    FFT1 = np.fft.fft(image * window, axis=1) / N   # Scaled FFT
    FFT2 = abs(FFT1[:,1:])**2                       # Square norm except DC

    if np.shape(FFT2)[1] % 2 == 0:
        FFT3 = 2 * FFT2[:, :len(FFT2)//2]           # multiply times 2...
        FFT3[:,-1] /= 2                             # ...except Nyquist
    else:
        FFT3 = 2 * FFT2[:, :int(len(FFT2)/2 +1)]
        
    NFFT = np.shape(FFT1)[1] + 0                    # !!! change to  np.shape(FFT3)[1]+1 ?    number of points in remaining FFT +1 for excluded DC
    delta_n = fs / NFFT                             # calculated bin width
    FFT4 = FFT3 / delta_n                           # final scaling
    
    if normalised  == True:
        psd1D_1 = FFT4  
    else:
        psd1D_1 = FFT4 * delta_n                    # if sum of output should yield sigma, undo scaling
    
    psd1D = np.nanmean(psd1D_1, axis = 0)           # average over all rows to get 1D PSD
    
    # to go from spatial wavelength spectra to temporal frequency spectra, divide by mean windspeed (Kaimal et al 1972)
    if normalised == True:
        psd1D /= np.mean(image)
    

    ####################################
    #### Plot with kolmogorov below ####
    ####################################
    
    x_axis = 2*pi*(1/((1/np.arange(1,np.shape(psd1D)[0]+1))*(2*samplerate*np.shape(psd1D)[0])))  # in radian?
    
    # Select kolmogorov drop off 
    begin = 0.003
    end = 0.01 # !!!   0.01 for 650m for 0.0063 for 1000m 
    
    axis_kolmogorov = x_axis[np.where((x_axis < end) & (x_axis > begin))]
    idx_kolmogorov = np.argmin(x_axis < end)
    value_kolmogorov = psd1D[idx_kolmogorov]
    a = (1/x_axis/2/pi)[idx_kolmogorov-len(axis_kolmogorov):idx_kolmogorov]**(5/3)
    kolmogorov = value_kolmogorov * a/ (min(a))
  
    if scaled == True:
        psd1D = x_axis*psd1D
    
    if plotting == True:
        plt.figure(figsize=(8,5))
        plt.loglog(x_axis[2:-1],psd1D[2:-1], linewidth=3, label = 'Power spectrum')
        plt.loglog(axis_kolmogorov, kolmogorov,'C3--',  linewidth=3, label = '-5/3')
        plt.title('One-sided PSD')
        plt.xlabel('$K\ [radians\ m^{-1}]$')
        plt.legend(ncol=1, loc='upper center', bbox_to_anchor=(0.8, 1), fontsize = 12)
        
        if scaled == True:
            plt.ylabel('K*S(K) [arbitrary]')
        if normalised == True:
            plt.ylabel(r'S(K) [$\frac{m^2}{s^2}/Hz$]')
        else:
            plt.title('One-sided PSD for variance')
            plt.ylabel(r'S(K) [$\frac{m^2}{s^2}$]')
    return psd1D


#%% find peak in psd1D

def sikora1997(windspeed, samplerate, window_length = 5, smoothing_fact = 3, windowed = True, plotting = False):
    from scipy.optimize import curve_fit
    from scipy import ndimage
    from scipy.signal import find_peaks

    # load the windpseed, detrend it and subtract mean
    detrended =  windspeed - trend(windspeed) 
    
    # calculate average 1D power spectrum, Hamming window applied on per row basis
    PSD = psd1D(detrended, samplerate, windowed = windowed, scaled = True, plotting = False)
    
    # apply smoothing on 1D spectrum
    smoothing_fact = smoothing_fact
    smoothed_spectrum = ndimage.gaussian_filter1d(PSD[0:], smoothing_fact) # 1.5
    
    # calculate x_axis in meters 
    x_axis = (1 / np.arange(1, np.shape(PSD)[0] + 1)) * (2 * samplerate * np.shape(PSD)[0]) 
    
    if plotting == True:
        plt.figure(figsize=(8,5))
        plt.title('detrended, windowed, averaged, along-wind psd1D', fontsize = 12)
        plt.loglog(x_axis, PSD[:],linewidth=3, label = 'original psd1D')
        plt.xlabel("$\lambda\ [m]$", fontsize = 12)
        plt.ylabel("$kS(k)$", fontsize = 12)
        plt.ylim((np.percentile(PSD[2:],0), 1.1 * np.percentile(PSD[2:],100)))
        plt.xlim(max(x_axis),min(x_axis))
        plt.tight_layout()
    
    # find spectral peaks if possible
    try:
        # select peaks in reasonable range for Zi to occur
        peaks, properties = find_peaks(np.squeeze(smoothed_spectrum), prominence=(None, 100000), distance=1)
        
        # selected largest peak (largest peak is not necessarily index that should be used for Zi!!!!)
        idx_Zi = peaks[np.argmax(smoothed_spectrum[peaks])] 

        # select lower limit of inertial subrange
        idx_ll = properties['right_bases'][np.argmax(smoothed_spectrum[peaks])]
        
        # calculate spatial wavelength corresponding to peak
        peak_spatial_wavelength = x_axis[idx_Zi]
        
        # determine Z_i paramater using ratio of 1.5
        Z_i = peak_spatial_wavelength / 1.5
        
        # following parameters are outdated and not used
        idx_ll = None                        # to be updated
        outdated1 = peaks                    # outdated
        outdated2 = idx_Zi                   # outdated
        
    except:
        print('Can\'t find a peak, consider changing window length')
        Z_i = np.NaN
        outdated1 = np.NaN
        outdated2 = np.NaN
        smoothed_spectrum = PSD
        idx_ll = None 
        
        pass
        
    return Z_i, outdated2, smoothed_spectrum, outdated1, x_axis, idx_ll, PSD

#%% first loop of Young's

def loop1(windfield):
    """
    First loop of Young's approach. Calculates surface stress Tau , friction velocity u* and roughness length z_0
    based on neutral wind speed input'
    
    input:
        windfield: 2D array with neutral 10 metre windspeeds in m/s
        
    output:
        surface_stress: surface stress field Tau
        friction velocity: u* in m/s
        friction length: z_0 in m
        neutral drag coefficient: C_dn
    """
    
    # define constants
    karman = 0.40                           # Karman constant
    Charnock = 0.011                        # Charnock constant
    g = 9.8                                 # Gravitational acceleration, m/s**2
    z = 12.5                                  # measurements height, 10 metres for CMOD5.N 
    rho_air = 1.2                           # kg/m**3, 1.2 for 20 degrees, 1.25 for 10 degreess                           
    T = 20                                  # temperature in Celcius
    
    # kinematic viscosity of air
    nu = 1.326 * 10**(-5) *(1 + (6.542 * 10**(-3))* T + (8.301 * 10**(-6)) * T**2 - (4.840 * 10**(-9)) * T**3) # m**2/s
    
    #Calculate mean neutral 10 metre windspeed
    windspeed_mean = np.mean(windfield)
    
    # prepare loop of 10 iterations
    iterations = 15
    A_friction_velocity = np.ones(iterations)    # m/s
    A_surface_stress = np.ones(iterations)       # kg/ m / s**2  [Pa]
    A_Cdn = np.ones(iterations)                  # 
    A_z_0 = np.ones(iterations)                  # m

    # Initialise loop with windspeed and iterate with refined estimates of neutral drag coefficient
    for i in range(iterations):
        if i > 0:
            A_friction_velocity[i] = np.sqrt(A_Cdn[i-1] * windspeed_mean**2)
            A_surface_stress[i] = rho_air * A_friction_velocity[i]**2
            A_z_0[i] = (Charnock * A_friction_velocity[i]**2) / g + 0.11 * nu / A_friction_velocity[i]
            A_Cdn[i] = (karman / np.log( z / A_z_0[i]) )**2
    
    # calculate stress field based on retrieved constants and windspeed estimates
    # !!! use mean windspeed here or windspeed? --> if use windspeed then u_star will change for different windspeed variances
    surface_stress = rho_air * A_Cdn[-1] * windfield**2 

    # save friction velocity and friction length based on mean stress field and neutral drag coefficient
    friction_velocity = np.sqrt(np.mean(surface_stress) / rho_air)
    z_0 = (Charnock * friction_velocity**2) / g + 0.11 * nu / friction_velocity
    Cdn = A_Cdn[-1]
    
    return surface_stress, friction_velocity, z_0, Cdn


#%% adapted first loop of Young's to recalculate wind field

def loop1_charnocks(windfield, Cdn0):
    """
    First loop of Young's approach. Calculates surface stress Tau , friction velocity u* and roughness length z_0
    based on neutral wind speed input'
    
    input:
        windfield: 2D array with neutral 10 metre windspeeds in m/s
        
    output:
        surface_stress: surface stress field Tau
        friction velocity: u* in m/s
        friction length: z_0 in m
        neutral drag coefficient: C_dn
    """
    
    # define constants
    karman = 0.40                           # Karman constant
    g = 9.8                                 # Gravitational acceleration, m/s**2
    z = 10                                  # measurements height, 10 metres for CMOD5.N 
    rho_air = 1.2                           # kg/m**3, 1.2 for 20 degrees, 1.25 for 10 degreess                           
    T = 20                                  # temperature in Celcius
    
    # calculate new Charnock constant as a range between 0.011 and 0.040 depending on gradient of windfield
    gradients = np.gradient(windfield)
    gradient  = np.sqrt(gradients[0]**2 + gradients[1]**2)
    # normalised_gradient = (gradient-np.nanmin(gradient)) / (np.nanmax(gradient)-np.nanmin(gradient))
    normalised_gradient = (windfield-np.nanmin(windfield)) / (np.nanmax(windfield)-np.nanmin(windfield))
    plottings(normalised_gradient)
    Charnock1 = normalised_gradient * (0.040-0.011) + 0.011
    plottings(Charnock1)
    # kinematic viscosity of air
    nu = 1.326 * 10**(-5) *(1 + (6.542 * 10**(-3))* T + (8.301 * 10**(-6)) * T**2 - (4.840 * 10**(-9)) * T**3) # m**2/s
    
    # prepare loop of 10 iterations
    iterations = 10
    A_friction_velocity = np.ones_like(windfield)    # m/s
    A_Cdn = np.ones_like(windfield)
    A_z_0 = np.ones_like(windfield)

    for i in range(iterations):
        if i > 0:
            A_friction_velocity = np.sqrt(A_Cdn * windfield**2)
            A_z_0 = (Charnock1 * A_friction_velocity**2) / g + 0.11 * nu / A_friction_velocity
            A_Cdn = (karman / np.log(z / A_z_0))**2
    
    Cdn0 = Cdn0
    Cdn = A_Cdn
    
    # calculate surface stress as using Cdn0 which uses Charnock= 0.011
    surface_stress = rho_air * Cdn0 * windfield**2 
    # recalculate wind field using Cdn which uses Charnock = [0.011, 0.040] as a function of wind speed slope
    windfield = np.sqrt(surface_stress / (rho_air * Cdn))
    surface_stress = rho_air * Cdn * windfield**2 
    
    # save friction velocity and friction length based on mean stress field and neutral drag coefficient
    friction_velocity = np.sqrt(np.mean(surface_stress) / rho_air)
    z_0 = np.mean((Charnock1 * friction_velocity**2) / g + 0.11 * nu / friction_velocity)
    Cdn = A_Cdn
    
    return surface_stress, friction_velocity, z_0, Cdn, windfield, Charnock1


#%% Loop 2A

def loop2A(windfield, surface_stress, friction_velocity, z_0, Zi):
    """
    Second loop of Young's approach. Requires output of previous loop. Recalculates wind field using stability correction.
    Outputs recalculated parameters, obukhov Length L and kinematic heat flux B
    
    input:
        windfield: 2D array with neutral 10 metre windspeeds in m/s
        surface_stress: Tau from loop 1
        friction_velocity: u_star from loop 1
        z_0: z_0 from loop 1
        Zi: lowest inversion height / CBL height following Kaimal et al 1976, Sikora et al 1997
    
    output:
        sigma_u: estimated wind field variance
        L: Obukhov length in meters
        B: Kinematic heat flux in metres
        w_star: convective velocity scale in m/s
        corr_fact: stability correction factor
    
    """
    
    z = 12.5                                  # measurements height, 10 metres for CMOD5.N 
    pi = 3.1415926535
    karman = 0.40                           # Karman constant
    rho_air = 1.2                           # kg/m**3, 1.2 for 20 degrees, 1.25 for 10 degreess 
    T_v = 298                               # virtual potential temperature in Kelvin
    g = 9.8                                 # Gravitational acceleration, m/s**2
    
    # prepare loop
    iterations = 10
    B_Cd = np.ones(iterations)
    B_Psi_m = np.ones(iterations)
    B_x = np.ones(iterations)
    B_L = np.array(np.ones(iterations))*np.inf*-1
    B_S = np.ones(iterations) * np.mean(windfield)
    B_sigma_u = np.ones(iterations)
    B_B = np.ones(iterations)
    
    for i in range(iterations):
        if i > 0:
            
            # Young et al 2000 and Paulson 1970
            B_x[i] = (1 + 16 * abs(z / B_L[i-1]))**0.25 
    
            # Young et al 2000 typo corrected
            B_Psi_m[i] = np.log(((1 + B_x[i]**2) / 2)**2) - 2 * np.arctan(B_x[i]) + pi / 2 
            
            # calculate drag coefficient
            B_Cd[i] = (karman / (np.log(z / z_0) - B_Psi_m[i]) )**2
            
            # recalculate wind field and compute average value
            recomp_windfield = np.sqrt(surface_stress / (B_Cd[i] * rho_air))
            B_S[i] = np.mean(recomp_windfield)

            # compute horizontal wind variance, alternatively derive this through spectrum and filter it
            B_sigma_u[i] =  np.std(recomp_windfield)

            # # Panofski et al 1977 eq 5
            B_L[i] = - Zi / (((B_sigma_u[i] / friction_velocity)**2 - 4) / 0.6)**(3/2) 
            
            # b = 0.73
            # c = 0.253
            # B_L[i] = - Zi*b**(3/2) / ((((B_sigma_u[i] / friction_velocity)**2 / (1 - (z/Zi)**c)) - 4))**(3/2) 
        
            # calculate kinematic heat flux
            B_B[i] = - (friction_velocity**3 * T_v) / (B_L[i] * karman * g)
    
    # calculate parameters to return as last iteration
    sigma_u = B_sigma_u[-1]
    L = B_L[-1]
    B = B_B[-1]
    w_star = friction_velocity*(-Zi/(karman * L))**(1/3)
    corr_fact = B_S[-1] / B_S[0]
    
    return sigma_u, L, B, w_star, corr_fact


#%% Loop 2B

def loop2B(windfield, friction_velocity, z_0, Zi, Cdn, PSD, samplerate, inertial_max, idx_ll, dissip_rate,  form = 'cells'):
    """
    Third loop of Young's approach. Requires output of loop 1. Recalculates wind field using stability correction.
    Similar to loop two but instead of using wind variance of entire field only uses inertial subrange
    Outputs recalculated parameters, obukhov Length L and kinematic heat flux B
    
    input:
        windfield: 2D array with neutral 10 metre windspeeds in m/s
        surface_stress: Tau from loop 1
        friction_velocity: u_star from loop 1
        z_0: z_0 from loop 1
        Zi: lowest inversion height following Kaimal et al 1976, Sikora et al 1997
        Cdn: neutral drag coefficient from loop 1
        PSD: 1D PSD from windspeed field following Stull 1988
        samplerate: windfield samplerate
        inertial_max: upper limit of inertial subrange
        idx_ll: inertial lower limit --> outdated
        form: expected convection form, cells is standard, rolls result in slight modification
        
    output:
        sigma_u: estimated wind field variance
        L: Obukhov length in meters
        B: Kinematic heat flux in metres
        w_star: convective velocity scale in m/s
        corr_fact: stability correction factor
    
    """
    
    # used for weighting w* values in inertial subrange
    def weighted_avg_and_std(values, weights):
        """
        Return the weighted average and weighted standard deviation.
        """
        average = np.average(values, weights=weights)
        variance = np.average((values-average)**2, weights=weights)
        return (average, np.sqrt(variance))
    
    
    pi = 3.1415926535
    z = 12.5                                 # measurements height, 10 metres for CMOD5.N 
    karman = 0.40                           # Karman constant
    T_v = 298                               # virtual potential temperature in Kelvin
    g = 9.8
    rho_air = 1.2
    Cp = 1005    
    
    #calculate x_axis of spatial wavenlengths
    x_axis = (1 / np.arange(1, np.shape(PSD)[0] + 1)) * (2 * samplerate * np.shape(PSD)[0])    # wavelengths 
    
    inertial_max = 6000                          # longest wavelength to consider for inertial subrange (Zi * 1.5)
    inertial_min = 5000  #x_axis[idx_ll]     # !!! shortest wavelength to consider for inertial subrange
    
    # select inertial subrange to be between inertial_min and inertial_max metres
    idx_kolmogorov = np.where((x_axis>inertial_min) & (x_axis < inertial_max))   
    iterations = 10
    kolmogorov = 0.5
    dissip_rate = dissip_rate       # 0.6 is low and 2 about average according to fig.4 in Kaimal et al. (1976)
    windspeed_mean = np.mean(windfield)
    
    # create arrays to store loop results
    C_w_star_normalised_deviation = np.ones(iterations)
    C_w_star = np.ones(iterations)
    C_B = np.ones(iterations)
    C_L = np.ones(iterations)
    C_epsilon = np.ones(iterations)
    C_dissip_rate = np.ones(iterations)
    C_x = np.ones(iterations)
    C_Psi_m = np.ones(iterations)
    C_corr_fact = np.ones(iterations)
    
    
    for i in range(iterations):
        if i > 0:
            # spatial wavelengths within selected part of inertial subrange
            Lambda = x_axis[idx_kolmogorov]

            # select PSD within inertial subrange and apply correction factor
            S = PSD[idx_kolmogorov] * C_corr_fact[i-1]**2
            
            # calcualte dimensionless frequency
            U_corr = windspeed_mean * C_corr_fact[i-1]
            n = 1/Lambda * U_corr
            fi = n * Zi / U_corr
        
            # Difference between 0.20 and 0.15 due to isotropy related to cross and along wind analysis (Kaimal et al 1976)
            # if analyses is performed cross wind (i.e. NOT cells), include 4/3 isotropy factor
            if form != 'cells':
                pre_w_star = np.sqrt((2 * pi)**(2/3) * fi**(2/3) * n * S / (4/3 * kolmogorov * dissip_rate**(2/3)))
                # pre_w_star = np.sqrt( (2 * pi * Zi / Lambda) * (2*pi/Lambda) * S / (4/3 * kolmogorov * dissip_rate**(2/3)))
            else:
                pre_w_star = np.sqrt((2 * pi)**(2/3) * fi**(2/3) * n * S / (kolmogorov * dissip_rate**(2/3)))
                # pre_w_star = np.sqrt( (2 * pi * Zi / Lambda) * (2*pi/Lambda) * S / (kolmogorov * dissip_rate**(2/3)))
            
            # determine weights and calculate weighted mean and std of convective velocity scale
            weights = x_axis[idx_kolmogorov]/np.min(x_axis[idx_kolmogorov])
            C_w_star[i] =  weighted_avg_and_std(pre_w_star, weights)[0]
            C_w_star_normalised_deviation[i] =  weighted_avg_and_std(pre_w_star, weights)[1] / np.median(pre_w_star)

            # calculate kinematic heat flux
            C_B[i] =  (C_w_star[i]**3 * T_v) / (g * Zi)
            
            # Monin Obukhov similarity theory
            C_L[i] = - (friction_velocity**3 * T_v) / (C_B[i] * karman * g)

            # structure function and emperical constant from young et al 2000
            C_x[i] = (1 + 16 * abs(z / C_L[i]))**0.25
            C_Psi_m[i] = np.log(((1 + C_x[i]**2) / 2)**2) - 2 * np.arctan(C_x[i]) + pi / 2 
        
            # stability correction factor from young et al 2000
            C_corr_fact[i] = 1 - (C_Psi_m[i] * np.sqrt(Cdn)) / karman
            
            
            
            # calculation of dissipation rate # TEST # 
            if form != 'cells':
                pre_epsilon = (2 * pi / U_corr) * (n**(5/3) * S / kolmogorov / (4/3) )**(3/2)
            else:
                pre_epsilon = (2 * pi / U_corr) * (n**(5/3) * S / kolmogorov)**(3/2)
            
            weights = x_axis[idx_kolmogorov]/np.min(x_axis[idx_kolmogorov])
            C_epsilon[i] = weighted_avg_and_std(pre_epsilon, weights)[0]
            C_dissip_rate[i] = - C_epsilon[i] * C_L[i] * karman / friction_velocity**3
            
        
    # calculate final outputs to return at the end of function
    sigma_u = friction_velocity * np.sqrt(4 + 0.6 * (-Zi / C_L[-1])**(2/3)) 
    L = C_L[-1]
    B = C_B[-1]
    w_star_normalised_deviation = C_w_star_normalised_deviation[-1]
    w_star = C_w_star[-1]
    corr_fact = C_corr_fact[-1]
    H = C_B[-1] * Cp * rho_air       # heat flux
    dissip_rate_test2 = C_dissip_rate[-1]

    return sigma_u, L, B, w_star, w_star_normalised_deviation, corr_fact, H , dissip_rate_test2


#%% Loop 2C using TKE dissipation

def loop2C(windfield, friction_velocity, Zi, Cdn, PSD, samplerate, inertial_max, idx_ll, form = 'cells'):
    
    # used for weighting w* values in inertial subrange
    def weighted_avg_and_std(values, weights):
        """
        Return the weighted average and weighted standard deviation.
        """
        average = np.average(values, weights=weights)
        variance = np.average((values-average)**2, weights=weights)
        return (average, np.sqrt(variance))
    
    
    pi = 3.1415926535
    z = 12.5                                 # measurements height, 10 metres for CMOD5.N 
    karman = 0.40                           # Karman constant
    iterations = 6                          # potentially increase this to 10?
    kolmogorov = 0.5
    T_v = 298                               # virtual potential temperature in Kelvin
    g = 9.8
    rho_air = 1.2
    Cp = 1005
    windspeed_mean = np.mean(windfield)
    
    #calculate x_axis of spatial wavenlengths
    x_axis = (1 / np.arange(1, np.shape(PSD)[0] + 1)) * (2 * samplerate * np.shape(PSD)[0])    # wavelengths 

    inertial_max =6000                           # longest wavelength to consider for inertial subrange (Zi * 1.5)
    inertial_min =5000 # !!! 650 #x_axis[idx_ll]      # shortest wavelength to consider for inertial subrange
    
    # select inertial subrange to be between inertial_min and inertial_max metres
    idx_kolmogorov = np.where((x_axis>inertial_min) & (x_axis < inertial_max))   

    S = PSD[idx_kolmogorov]
            
    # create arrays to store loop results
    D_epsilon = np.ones(iterations)
    D_epsilon_std = np.ones(iterations)
    D_L = np.ones(iterations)
    D_B = np.ones(iterations)
    D_x = np.ones(iterations)
    D_Psi_m = np.ones(iterations)
    D_corr_fact = np.ones(iterations)
    
    for i in range(iterations):
        if i > 0:
            
            U_corr = windspeed_mean * D_corr_fact[i-1]
            S = PSD[idx_kolmogorov] * D_corr_fact[i-1]**2
            Lambda = x_axis[idx_kolmogorov]
            n = 1/Lambda * U_corr 
            
            # if not cells, analysis is performed cross wind, required 4/3 isotropy correction
            if form != 'cells':
                pre_epsilon = (2 * pi / U_corr) * (n**(5/3) * S / kolmogorov / (4/3) )**(3/2)
            else:
                pre_epsilon = (2 * pi / U_corr) * (n**(5/3) * S / kolmogorov)**(3/2)
            
            weights = x_axis[idx_kolmogorov]/np.min(x_axis[idx_kolmogorov])
            D_epsilon[i] = weighted_avg_and_std(pre_epsilon, weights)[0]
            D_epsilon_std[i] =  weighted_avg_and_std(pre_epsilon, weights)[1]
            
            Y = np.arange(-0.00001,-100,-0.00005)  # Y = z/ L
            c1 = 0.88; c2 = 2.06; # Kooijmans and Hartogenis (2016)
            lhs = c1 * ((1 - c2 * (Y))**(-1/4) - Y)
            rhs = D_epsilon[i] * karman * z / friction_velocity**3 
            
            # find index with minimum difference between approaches
            idx_same = (np.abs(lhs - rhs)).argmin()
            D_L[i] = z / Y[idx_same]
            
            D_B[i] = - (friction_velocity**3 * T_v) / (D_L[i] * karman * g)
            
            # Calculate correction for wind speed based on stability
            D_x[i] = (1 + 16 * abs(z / D_L[i]))**(1/4)
            D_Psi_m[i] = np.log(((1 + D_x[i]**2) / 2)**2) - 2 * np.arctan(D_x[i]) + pi / 2 
        
            # stability correction factor from young et al 2000
            D_corr_fact[i] = 1 - (D_Psi_m[i] * np.sqrt(Cdn)) / karman
    
    # if L approaches near neutral values it it probably fell off the graph
    L = D_L[-1]
    if abs(L) >= 9999:
        L = np.nan
        
    # calculate final outputs to return at the end of function    
    sigma_u = friction_velocity * np.sqrt(4 + 0.6 * (-Zi / L)**(2/3)) 
    D_epsilon = D_epsilon[-1]
    D_epsilon_std = D_epsilon_std[-1]
    corr_fact = rhs# D_corr_fact[-1]
    H = D_B[-1] * Cp * rho_air       # heat flux
    
    return sigma_u, L, D_epsilon, D_epsilon_std, corr_fact, H


#%% Apply all calculates on per tile basis

def tiledWind(NRCS, incidence, longitude, latitude, iterations, size, dissip_rate, slope_multiplier = None, \
              w_star_threshold = None, Zi_input = None, True_wind = None, form = 'cells', samplerate = 50, plotting = True):
    from scipy import ndimage
    from tensorflow import keras 
    
    #load classification model
    saved_model = keras.models.load_model("model_v4.h4")
    
    """
    This function performs the encessary calculations to characterise the MABL 
    
    input:
        NRCS input: 2D radar image
        incidence: 2D image containing incidence angle corresponding to NRCS input
        longitude: 2D image containing longitude corresponding to NRCS input
        latitude: 2D image containing latitude corresponding to NRCS input
        iterations:  number of iterations for CMOD algorithm
        size: size of tiles in pixels
        dissip_rate: dissipation rate for loop 2B, approxiamntely between 0.5 and 2.5 (kaimal et al,  1976)
        slope_multiplier: multiplier for wind speed field slope, higher slope leads to more variance
        w_star_threshold: quality threshold for w_star parameter, usually between 0.02 and 0.04
        Zi_input: a-priori value for Z_i, if not submitted this value will be calculated (less accurate)
        True_wind: a-priori value for wind direction, if not submitted this value will be calculated (less accurate)
        form: convection type present in scene, either 'cells' or anything else
        samplerate: NRCS sampelrate in metres
        plotting: whether to plot grid on NRCS
        
    output:
        all parameters from loop 1, 2A, 2B and 2C
    
    """
            
    pi = 3.1415926535
    
    # determine how many tiles to make
    M = N = size                                # size of tile
    im = np.where(NRCS==0, np.nan, NRCS)        # set 0 values (no data) to nan
    inc = incidence
    sizeY  = im.shape[0]//M
    sizeX  = im.shape[1]//N
    
    # function that tiles the images, specifically for lat and long(cant recall other purpose)
    def tile(image, shape1, shape2):
        tiledPre = [image[x:x+shape1,y:y+shape2] for x in range(0,image.shape[0],shape1) for y in range(0,image.shape[1],shape2)]
        idxKeep = np.squeeze(np.array(np.where([np.shape(i)==(shape1,shape2) for i in tiledPre])))
        tile2Keep = [tiledPre[i] for i in idxKeep]
        return tiledPre, idxKeep, tile2Keep
    
    # calculates mean lat and long of tiles for later use in plotting
    _, _,tile2Keep_lon = tile(longitude, M, N)
    _, _,tile2Keep_lat = tile(latitude, M, N)
    mean_lon = np.array([np.mean(tile2Keep_lon[i]) for i in range(0, len(tile2Keep_lon))])
    mean_lat = np.array([np.mean(tile2Keep_lat[i]) for i in range(0, len(tile2Keep_lat))])
    
    # if input is true wind array (with pixels of 1 by 1km) then it calculates the mean wind direction within 25 by 25km
    if type(True_wind) == np.ndarray:
        M2 = 25; N2 = 25
        _, _,tile2Keep_lon = tile(True_wind, M2, N2)
        mean_wind_direction = np.array([np.mean(tile2Keep_lon[i]) for i in range(0, len(tile2Keep_lon))])
        
    # subdivide radar, incidence and lat and lon images into equal sized tiles
    tiles = [im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]
    idxKeep_tiles = np.squeeze(np.array(np.where([np.shape(i)==(M,N) for i in tiles])))

    tiles_incidence = [inc[x:x+M,y:y+N] for x in range(0,inc.shape[0],M) for y in range(0,inc.shape[1],N)]
    idxKeep_tiles_incidence = np.squeeze(np.array(np.where([np.shape(i)==(M,N) for i in tiles_incidence])))
    
    tiles_lat = [latitude[x:x+M,y:y+N] for x in range(0,latitude.shape[0],M) for y in range(0,latitude.shape[1],N)]   
    tiles_lon = [longitude[x:x+M,y:y+N] for x in range(0,longitude.shape[0],M) for y in range(0,longitude.shape[1],N)]

    # calculate azimuth angle based on coordinates
    rangeLookAngle = calculate_initial_compass_bearing((latitude[0,0],longitude[0,0]),(latitude[0,-1],longitude[0,-1]))
    azimuthAngle = rangeLookAngle - 90
    
    # Prepare lists for saving results
    hold_PSD = []
    hold_prediction = np.ones(len(idxKeep_tiles))
    hold_form = np.zeros(len(idxKeep_tiles))
    hold_angle = np.zeros(len(idxKeep_tiles))
    hold_u_star = np.zeros(len(idxKeep_tiles))
    hold_z_0 = np.zeros(len(idxKeep_tiles))
    hold_Cdn = np.zeros(len(idxKeep_tiles))
    hold_sigma_u1 = np.zeros(len(idxKeep_tiles))
    hold_sigma_u2 = np.zeros(len(idxKeep_tiles))
    hold_L1 = np.zeros(len(idxKeep_tiles))
    hold_L2 = np.zeros(len(idxKeep_tiles))
    hold_dissip_rate = np.zeros(len(idxKeep_tiles))
    hold_L3 = np.zeros(len(idxKeep_tiles))
    hold_PSD_max = np.zeros(len(idxKeep_tiles))
    hold_w_star2 = np.zeros(len(idxKeep_tiles))*np.nan
    hold_w_star2_std = np.zeros(len(idxKeep_tiles))*np.nan
    hold_Zi = np.zeros(len(idxKeep_tiles))
    hold_corr_fact1 = np.zeros(len(idxKeep_tiles))
    hold_corr_fact2 = np.zeros(len(idxKeep_tiles))
    hold_H = np.zeros(len(idxKeep_tiles))
    hold_windspeed = np.zeros(len(idxKeep_tiles))
    hold_epsilon= np.zeros(len(idxKeep_tiles))
    
    # perform calculations on each tile
    figall, axes = plt.subplots(nrows=sizeY, ncols=sizeX, figsize=(sizeY*3 , sizeX*1.5), sharex=True, sharey=True)
    
    # flip axis such that it alligns with imshow plots
    for i, ax in enumerate(np.flipud(np.fliplr(axes).flat)):
    
        # select tile
        item = np.array(tiles[idxKeep_tiles[i]])
        item_incidence = np.array(tiles_incidence[idxKeep_tiles_incidence[i]])
        
        # if tile is full (i.e. not a clipped border tile), commence calculation
        if np.shape(item)[0] == M & np.shape(item)[1] == N:
           try:
               # load image
               image = item
               image_incidence = item_incidence
               
               # append nan to list just to fill index, will be overwritten if PSD is succesfully calculated
               hold_PSD.append(np.nan)
               
               """
               0. classify image as roll or cell, only to be used for 300x300 metre pixels
               """
               
               filt = image[26:, 26:]
               filt_norm = (filt-np.percentile(filt,1))/(np.percentile(filt,99)-np.percentile(filt,1))
               
               # 57 is harcoded dimension of training data
               filt_norm_resampled = ndimage.zoom(filt_norm, 57/filt_norm.shape[1], order=0)
               filt_norm_resampled_reshaped = np.reshape(filt_norm_resampled, (1, filt_norm_resampled.shape[1], filt_norm_resampled.shape[1], 1))
               
               # predict class based on loaded home brew model
               prediction = saved_model.predict(filt_norm_resampled_reshaped)
               hold_prediction[i] = np.max(prediction)
               hold_form[i] = np.argmax(prediction)
               
               label_names = ['cells' , 'rolls']
               # uncomment next line if want to use CNN classification for convection type 
               # form = label_names[np.argmax(prediction)]
            
               """
               1. determine wind direction angle either through peak or gaussian
               """
               # Filter image by dividing by longwave components and then apply Hamming window
               image_longwave = longWaveFilter(item, samplerate)
               hamming_window = HammingWindow(item)
               image_filtered = (image-image_longwave)/image_longwave*hamming_window

               # calculate 2D PSD for resulting image and apply Gaussian filter to Power Spectrum for smoothing
               psd2D = twoDPS(image_filtered, samplerate)
               psd2D_filtered = ndimage.gaussian_filter(psd2D,3)

               # calculate angle in spectrum and relate that to wind origin, if rolls use 90 degree offset
               angle, _, _ = peakShortwave(psd2D_filtered, samplerate, False)
               
               # Apply correction for use of rolls over cells
               if form != 'cells':
                   rolloffset = 20  # (faller 1964 said 13 degrees, wackerman between 13 and 30 degrees --> will choose 20 degrees)
                   angle -= (90 - rolloffset) ; 
               
               # calcualte wind direction a.k.a. wind origin
               wind_origin = (azimuthAngle - angle) - 180
               if wind_origin <= 0:
                   wind_origin += 360
               
               # if single wind speed direction is given, use that, if array is given  respective array index
               if (type(True_wind) == int) or (type(True_wind) == float):
                   wind_origin = True_wind
               elif type(True_wind) == np.ndarray:
                   wind_origin = mean_wind_direction[i]

               """
               2. rotate radar image and incidence angle based on found wind angle
               """
               
               # -1 to account for rotation direction of np.rotate, then times np.sign() to account for loss of sign due to abs()
               rotation = (abs(wind_origin - rangeLookAngle) % 180) * -1 * np.sign(wind_origin - rangeLookAngle)
               
               # undo roll correction to rotate in axis greatest variability
               if form != 'cells':
                   rotation -= (90 - rolloffset) 
               
               # rotate image such that horizontal is greatest variability
               image_rotated = rotateAndClip(image, rotation, samplerate)
               incident_angle_rotated = rotateAndClip(image_incidence, rotation, samplerate)

               """
               3. compute windspeed with CMOD and determine peak in along- or cross-wind psd1D
               """
               
               # Calculate wind speed and along wind psd1D, optionally used CMOD IFR2
               windspeed = applyCMOD(image_rotated, wind_origin - rangeLookAngle, incident_angle_rotated, iterations, samplerate, CMOD5 = False, plotting = False)
               # windspeed = applyCMOD_IFR2(image_rotated, wind_origin - rangeLookAngle, incident_angle_rotated, iterations, samplerate, plotting = False)
               
               # multiply wind field variance to assess influence (OPTIONAL)
               slope_multiplier = slope_multiplier # !!! default is 1.0
               windspeed = (windspeed - np.mean(windspeed)) * slope_multiplier + np.mean(windspeed)
               
               # determine peak Zi using method described in Sikora 1997
               try: 
                   Zi_sikora, powerlaw, smoothed_spectrum, peak_idx, _, idx_ll, PSD_sikora = sikora1997(windspeed, samplerate, window_length = 7, \
                                                                          smoothing_fact = 2, windowed = True, plotting = False)
               except:
                   print('Sikora failed')
                   Zi_sikora = np.nan()
               
               # correct for aspect ratio of convection is not structured as cells
               if form != 'cells':
                   Zi_ratio = 2.0 # average ratio of 3.39 according to wang 2019 "characeristics of marine atmospheric boundary layer"
               else:
                   Zi_ratio = 1.5  # kaimal 1976 and sikora 1997 
               
               # if value of Zi is pre specified, use that instead
               if (type(Zi_input) == int) or (type(Zi_input) == float):
                   Zi = Zi_input
               else:
                   # undo ratio of 1.5 in sikora function and apply new ratio
                   Zi_sikora = Zi_sikora * 1.5 / Zi_ratio
                   Zi = Zi_sikora
               
               """
               4. calculate atmospheric parameters
               """
               # loop 1
               tau, u_star, z_0, Cdn = loop1(windspeed)
               # tau, u_star, z_0, Cdn, windspeed = loop1_charnocks(windspeed, Cdn)
               
               # loop 2A
               sigma_u1, L1, B1, w_star1, corr_fact1 = loop2A(windspeed, tau, u_star, z_0, Zi)
               
               # loop 2B
               # set maximum values used in inertial subrange approach to the peak (e.g. Zi * 1.5)
               inertial_max = Zi * Zi_ratio
               
               # calculate temporal PSD (F(u) to S(u))
               PSD_original = psd1D(windspeed, samplerate, plotting = False, windowed = True, scaled = False, normalised = True)
               
               # PSD variance multiplier to test effect (OPTIONAL)
               PSD_multiplier = 1 # !!! default is 1.0
               PSD_original = PSD_original * PSD_multiplier
               
               #account for form (e.g. cells vs rolls)
               sigma_u2, L2, B2, w_star2, w_star2_std, corr_fact2, H, dissip_rate_test2 = loop2B(windspeed, u_star, z_0, Zi, Cdn, PSD_original, samplerate, inertial_max, idx_ll, dissip_rate, form = form) 
               
               # Loop 2C
               sigma_u3, L3, D_epsilon, D_epsilon_std, corr_fact3, H3 = loop2C(windspeed, u_star, Zi, Cdn, PSD_original, samplerate, inertial_max, idx_ll, form = form)
               hold_epsilon[i] = D_epsilon
               
               """
               5. Save parameters for which inertial subrange has sufficient quality, i.e. w* meets threshold
               """
               
               # use smoothed spectrum for plotting normalised with wavelength (smoothed_spectrum from sikora) or original PSD
               if np.isnan(Zi_sikora) == False:
                   # PSD = smoothed_spectrum
                   # slope_power = 2/3
                   slope_label = '-2/3'
                   fact0 = 1
               else:
                   # PSD = PSD_original
                   # slope_power = 5/3
                   slope_label = '-5/3'
                   fact0 = np.nan
                   
               # if threshold is not set, ignore QC by setting to infinity
               if (type(w_star_threshold) == int) or (type(w_star_threshold) == float):
                   threshold = w_star_threshold
               else:
                   threshold = np.inf
               
               if (w_star2_std >= threshold) or (w_star2_std <= 0.003):
                   fact1 = np.nan
               else:
                   fact1 = 1
                   
               # optionally filter on quality of classification e.g. all values below 0.8 filter out
               if (hold_prediction[i] <= 0.00) :
                   fact2 = np.nan
               else:
                   fact2 = 1

               # store parameters unless quality thresholds not met, then store NaN
               hold_PSD[i] = PSD_original * fact0 * fact1 * fact2 
               hold_angle[i] = wind_origin  * fact0 * fact1 * fact2
               hold_u_star[i] = u_star  * fact0 * fact1 * fact2
               hold_z_0[i] = z_0  * fact0 * fact1 * fact2
               hold_Cdn[i] = Cdn  * fact0 * fact1 * fact2
               hold_sigma_u1[i] = sigma_u1  * fact0 * fact1  * fact2
               hold_sigma_u2[i] = sigma_u2  * fact0 * fact1 * fact2
               hold_L1[i] = L1  * fact0 * fact1  * fact2
               hold_L2[i] = L2  * fact0 * fact1  * fact2
               hold_dissip_rate[i] = dissip_rate_test2  * fact0 * fact1  * fact2
               hold_L3[i] = L3  * fact0 * fact1  * fact2
               hold_PSD_max[i] = np.max(PSD_original[2:]) * fact0 * fact1  * fact2
               hold_w_star2[i] = w_star2
               hold_w_star2_std[i] = w_star2_std
               hold_Zi[i] = Zi  * fact0 * fact1   * fact2
               hold_corr_fact1[i] = corr_fact1  * fact0 * fact1  * fact2
               hold_corr_fact2[i] = corr_fact2  * fact0 * fact1   * fact2
               hold_H[i] = H  * fact0 * fact1   * fact2
               hold_windspeed[i] = np.mean(windspeed)  * fact0 * fact1  * fact2
               
           except:
               pass
    
    """
    6. plot results per tile
    """
    # delete anomalous high S(n) values
    fact3 = np.where(hold_PSD_max > 2 * np.nanmedian(hold_PSD_max[hold_PSD_max!=0]), np.nan, 1)
    
    # plot and filter resuls based on ensemble statistics
    for i, ax in enumerate(np.flipud(np.fliplr(axes).flat)):
        # select tile
        item = np.array(tiles[idxKeep_tiles[i]])
        
        # delete scenes containing nan radar values
        fact4 = np.mean(item) / np.mean(item)
        
        # store parameters unless quality thresholds not met, then store NaN
        hold_angle[i] = hold_angle[i] * fact3[i] * fact4
        hold_u_star[i] = hold_u_star[i] * fact3[i] * fact4
        hold_z_0[i] = hold_z_0[i] * fact3[i] * fact4
        hold_Cdn[i] = hold_Cdn[i] * fact3[i] * fact4
        hold_sigma_u1[i] = hold_sigma_u1[i] * fact3[i] * fact4
        hold_sigma_u2[i] = hold_sigma_u2[i] * fact3[i] * fact4
        hold_L1[i] = hold_L1[i] * fact3[i] * fact4
        hold_L2[i] = hold_L2[i] * fact3[i] * fact4
        hold_dissip_rate[i] = hold_dissip_rate[i]  * fact3[i] * fact4
        hold_L3[i] = hold_L3[i] * fact3[i] * fact4
        hold_PSD_max[i] = hold_PSD_max[i] * fact3[i] * fact4
        hold_w_star2[i] = hold_w_star2[i] * fact3[i] * fact4
        hold_w_star2_std[i] = hold_w_star2_std[i] * fact3[i] * fact4
        hold_Zi[i] = hold_Zi[i] * fact3[i] * fact4
        hold_corr_fact1[i] = hold_corr_fact1[i] * fact3[i] * fact4
        hold_corr_fact2[i] = hold_corr_fact2[i] * fact3[i] * fact4
        hold_H[i] = hold_H[i] * fact3[i] * fact4
        hold_windspeed[i] = hold_windspeed[i] * fact3[i] * fact4
    
        # zero values mean no calculationw as performed, set to nan, (alternatively fill initial list with nan's instead of zeros)
        hold_Zi = np.where(hold_Zi == 0, np.nan, hold_Zi)
        hold_sigma_u2 = np.where(hold_sigma_u2 == 0, np.nan, hold_sigma_u2)
        hold_u_star = np.where(hold_u_star == 0, np.nan, hold_u_star)
        hold_L2 = np.where(hold_L2 == 0, np.nan, hold_L2)  
        hold_L3 = np.where(hold_L3 == 0, np.nan, hold_L3)  
        hold_windspeed = np.where(hold_windspeed == 0, np.nan, hold_windspeed)    
    
        size_plot = samplerate*size/1000
    
        # print(hold_Zi)
        fact5 = hold_sigma_u2 / hold_sigma_u2
        
        if np.shape(item)[0] == M & np.shape(item)[1] == N:
           try:
               PSD_plot = hold_PSD[i] * fact5[i]
             
               # if PSD was filetred out (i.e. has NaN value) then skip plotting
               if np.isnan(PSD_plot).any():
                   pass
               
               else:
                   # calculate x_axis for plotting
                   x_axis = 2*pi / ((1/np.arange(1,np.shape(PSD_plot)[0]+1))*(2*samplerate*np.shape(PSD_plot)[0]))

                   # Select kolmogorov begin and end point for plotting
                   begin = 2 * pi / (hold_Zi[i] * Zi_ratio)   # 0.0042

                   # always start inertial subrange for plotting  from 650 spatial wavelength or set to x_axis[idx_ll]
                   end = 0.009667 

                   # select indexes of values for plotting of klmogorov slope
                   axis_kolmogorov = x_axis[np.where((x_axis < end) & (x_axis > begin))]
                   idx_kolmogorov = np.argmin(x_axis < end) #np.where((x_axis > 600) & (x_axis < 3000))])
                   value_kolmogorov = PSD_plot[idx_kolmogorov]
               
                   # slope either 5/3 or 2/3 depending on whether its scaled by x-axis or not
                   a = (1/x_axis/2/pi)[idx_kolmogorov-len(axis_kolmogorov):idx_kolmogorov]**(5/3)
                   kolmogorov = value_kolmogorov * a/ (min(a))
               
                   # plot results
                   ax.loglog(x_axis[2:], PSD_plot[2:], color = 'C0', label = '')
                   ax.loglog(axis_kolmogorov, kolmogorov,'C3--',  linewidth=3, label = slope_label)
                   # ax.scatter(x_axis[max_peak], PSD_plot[max_peak], s=40, c= 'k')
                   ax.set_ylim(kolmogorov[-1] / 5, kolmogorov[0] * 18)

                   # add statistics to plots
                   textstr0 = 'Passed' 
                   # ax.text(0.05, 0.05, textstr0, color = 'g', transform= ax.transAxes, fontsize=20, horizontalalignment='left', verticalalignment='bottom')
                   ax.text(0.05, 0.9, i+1, color = 'r',transform= ax.transAxes, fontsize=20, verticalalignment='top')

               completention = i/(sizeY*sizeX)*100
               # print('Completention percentage: %1.0f' %completention)
           except:
               pass    

    # add image containing location of grids
    if plotting == True:
        gridded(longitude, latitude, NRCS, tiles_lon, tiles_lat, idxKeep_tiles, mean_lat, mean_lon, size_plot)
    
    figall.text(0.5, 0.92, 'PSD for area %1.1f$^2$ km$^2$' %size_plot, ha='center', va='center', fontsize=5*sizeY)     
    figall.text(0.5, 0.08, r'Spatial wavenumber $k$ $[rad\ m^{-1}]$', ha='center', va='center', fontsize=4*sizeY)
    figall.text(0.08, 0.5, r'PSD $[m^{2}s^{-2}Hz^{-1}]$', ha='center', va='center', rotation='vertical', fontsize=4*sizeY)
    
    return hold_angle, mean_lon, mean_lat, hold_u_star, hold_z_0, hold_sigma_u1, hold_sigma_u2, \
        hold_L1, hold_L2, hold_L3, hold_w_star2, hold_w_star2_std, hold_Zi, hold_corr_fact1, hold_corr_fact2, \
            hold_windspeed, tiles_lon, tiles_lat, idxKeep_tiles, hold_prediction, hold_form, hold_epsilon, \
                hold_H, hold_dissip_rate
   
    


