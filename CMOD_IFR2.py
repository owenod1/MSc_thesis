# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 17:11:33 2021

@author: Owen O'Driscoll
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 12:12:44 2021

@author: Owen O'Driscoll
"""

# %reset -f

import warnings
import numpy as np
from matplotlib import pyplot as plt  
import netCDF4 as nc


"""
skeleton code retrieved from 
https://gitlab.tudelft.nl/drama/stereoid/-/blob/a8a31da38369a326a5172d76a51d73bba8bc2d58/stereoid/oceans/cmod5n.py

model retrieved from 
Quilfen et al. 1998,  observation of tropical cyclones by high-resolution scatterometry.

"""

# Ignore overflow errors for wind calculations over land
warnings.simplefilter("ignore", RuntimeWarning)

def cmodIFR2_forward(v, phi, theta):
    """!     ---------
    !     cmodIFR2_forward(v, phi, theta)
    !         inputs:
    !              v     in [m/s] wind velocity (always >= 0)
    !              phi   in [deg] angle between look direction and wind direction
    !              theta in [deg] incidence angle
    !         output:
    !              CMOD IFR2 BACKSCATTER (LINEAR)
    !
    !        All inputs must be Numpy arrays of equal sizes
    
            Source:
                Quilfen, Y., Chapron, B., Elfouhaily, T., Katsaros, K., & Tournadre, J. (1998). 
                Observation of tropical cyclones by high‐resolution scatterometry. Journal of 
                Geophysical Research: Oceans, 103(C4), 7767-7786.
    !---------------------------------------------------------------------
       """
       
    # NB: 0 added as first element below, to avoid switching from
    # 1-indexing to 0-indexing
    C = [              # CMOD IFR2
        0,
        -2.437597,
        -1.567031,
        0.370824,
        -0.040590,
        0.404678,
        0.188397,
        -0.027262,
        0.064650,
        0.054500,
        0.086350,
        0.055100,
        -0.058450,
        -0.096100,
        0.412754,
        0.121785,
        -0.024333,
        0.072163,
        -0.062954,
        0.015958,
        -0.069514,
        -0.062945,
        0.035538,
        0.023049,
        0.074654,
        -0.014713,
    ]
    
    pi = 3.1415926535
    fi = np.radians(phi)
    V = v
    
    x1 = (theta - 36) / 19
    P1_1 = x1
    P1_2 = (3 * x1**2 - 1) / 2
    P1_3 = x1 * (5 * x1**2 - 3) / 2
    
    Alpha = C[1] + C[2] * P1_1 + C[3] * P1_2 + C[4] * P1_3
    Beta = C[5] + C[6] * P1_1 + C[7] * P1_2
    
    
    x2 = (2 * theta - 76) / 40
    P2_1 = x2
    P2_2 = 2 * x2**2 -1
    
    V1 = (2 * V - 28) / 22
    V2 = 2 * V1**2 - 1
    V3 = (2 * V2 - 1) * V1
    
    B1 = C[8] + C[9]*V1 + C[10]*P2_1 + C[11]*P2_2*V1 + C[12]*P2_2 + C[13]*P2_2*V1
    
    B2 = C[14] + C[15]*P2_1 + C[16]*P2_2 + \
        (C[17] + C[18]*P2_1 + C[19]*P2_2) * V1 +\
        (C[20] + C[21]*P2_1 + C[22]*P2_2) * V2 +\
        (C[23] + C[24]*P2_1 + C[25]*P2_2) * V3      
        

    ######### TANH in radians ###########
    Sigma = (10**(Alpha + Beta * np.sqrt(V))) * (1 + B1 * np.cos(fi) + \
            np.tanh(B2) * np.cos(2*fi))  
    
    return Sigma


def cmodIFR2_inverse(sigma0_obs, phi, incidence, iterations=10):
    """!     ---------
    !     cmod5n_inverse(sigma0_obs, phi, incidence, iterations)
    !         inputs:
    !              sigma0_obs     Normalized Radar Cross Section [linear units]
    !              phi   in [deg] angle between azimuth and wind direction
    !                    (= D - AZM)
    !              incidence in [deg] incidence angle
    !              iterations: number of iterations to run
    !         output:
    !              Wind speed, 10 m, neutral stratification
    !
    !        All inputs must be Numpy arrays of equal sizes
    !
    !    This function iterates the forward CMOD5N function
    !    until agreement with input (observed) sigma0 values
    !---------------------------------------------------------------------
       """
    from numpy import ones, array

    # First guess wind speed
    V = array([10.0]) * ones(sigma0_obs.shape)
    step = 10.0

    # Iterating until error is smaller than threshold
    for iterno in range(1, iterations):
        # print(iterno)
        sigma0_calc = cmodIFR2_forward(V, phi, incidence)
        ind = sigma0_calc - sigma0_obs > 0
        V = V + step
        V[ind] = V[ind] - 2 * step
        step = step / 2
        
    Bias_1 = 0.0831 * V - 0.0173 * V**2 + 0.0009 * V**3
    Bias_2 = np.arctan(V - 22) + 3.0382

        
    V = np.where((V > 10) & (V <= 22), V + Bias_1, V)
    V = np.where((V > 22), V + Bias_2, V)
        
    return V


# # to test the functioning
# test_NRC = 10**(np.array([[-31.84, -32.81, -14.45, -19.02, -2.57], [ -2.71, 4.38, 5.32, 4.74, 6.19]])/10)
# test_Phi = np.array([[0 , 90, 0 , 90, 0], [ 180, 0, 180, 0, 180]])
# test_theta = np.array([[60 , 60, 40, 40, 25],[ 25, 18, 18, 18, 18]])
# test_V = np.array([[1 , 1, 8 , 8 ,15 ],[ 15, 22, 22, 28, 28]])
# iterations = 10

# windspeed = cmodIFR2_inverse(test_NRC, test_Phi, test_theta, iterations = iterations)
# test = 10*np.log10(cmodIFR2_forward(test_V,test_Phi, test_theta ))
