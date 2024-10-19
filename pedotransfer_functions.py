import numpy as np

"""

wc is the water content at a given matric potential (cm3
/cm3
)
silt is the percentage silt (%)
clay is the percentage clay (%)
oc is the percentage organic carbon (%)
BD is the bulk density (g/cm3
)
t is a qualitative variable having the value of 1 


mbar is matric potential (cm), which is 50 at field capacity, 1000 at 1 bar, 15,000 at 15 bar or wilting point,
and 1,000,000 at 1,000 bar

To convert from water content to soil moisture deficit (mm) used by RothC the following equation is used.
Mi= (WCi-WCfc)*10*depth,
Where:
Mi is the soil moisture deficit at Mb, M or Mc
WCfc is the water content at field capacity
WCi is the water content at -1 bar, -15 bar, or -1000 bar 
"""

def calc_mbar( x):
    return -1000.0 * x
import numpy as np

def calculate_alpha(clay, silt, oc, bd, t):
    """
    Calculate the parameter alpha using the given formula.
    
    Parameters:
    - clay: percentage of clay
    - silt: percentage of silt
    - oc: organic carbon content (percentage)
    - bd: bulk density (g/cm³)
    - t: temperature (°C)
    
    Returns:
    - alpha: calculated parameter alpha
    """
    oc1_72 = oc * 1.72
    
    alpha = np.exp(
        -14.96 +
        0.03135 * clay +
        0.0351 * silt +
        0.646 * oc1_72 +
        15.29 * bd -
        0.192 * t -
        4.671 * bd**2 -
        0.000781 * clay**2 -
        0.00687 * oc1_72**2 +
        0.0449 / oc1_72 +
        0.0663 * np.log(silt) +
        0.1482 * np.log(oc1_72) -
        0.04546 * bd * silt -
        0.4852 * bd * oc1_72 +
        0.00673 * clay * t
    )
    
    return alpha

def calculate_theta_s(clay, silt, oc, bd, t):
    """
    Calculate the parameter theta_s using the given formula.
    
    Parameters:
    - clay: percentage of clay
    - silt: percentage of silt
    - oc: organic carbon content (percentage)
    - bd: bulk density (g/cm³)
    - t: temperature (°C)
    
    Returns:
    - theta_s: calculated parameter theta_s
    """
    oc1_72 = oc * 1.72
    
    theta_s = (
        0.7919 +
        0.001691 * clay -
        0.29619 * bd -
        0.000001491 * silt**2 +
        0.0000821 * oc1_72**2 +
        0.02427 / clay +
        0.01113 / silt +
        0.01472 * np.log(silt) -
        0.0000733 * oc1_72 * clay -
        0.000619 * bd * clay -
        0.001183 * bd * oc1_72 -
        0.0001664 * silt * t
    )
    
    return theta_s

def calculate_theta_r():
    """
    Calculate the parameter theta_r.
    
    Returns:
    - theta_r: constant value of theta_r
    """
    theta_r = 0.01
    return theta_r

def calculate_n(clay, silt, oc, bd, t):
    """
    Calculate the parameter n using the given formula.
    
    Parameters:
    - clay: percentage of clay
    - silt: percentage of silt
    - oc: organic carbon content (percentage)
    - bd: bulk density (g/cm³)
    - t: temperature (°C)
    
    Returns:
    - n: calculated parameter n
    """
    oc1_72 = oc * 1.72
    
    n = np.exp(
        -25.23 -
        0.02195 * clay +
        0.0074 * silt -
        0.194 * oc1_72 +
        45.5 * bd -
        7.24 * bd**2 +
        0.0003658 * clay**2 +
        0.002885 * oc1_72**2 -
        12.81 / bd -
        0.1524 / silt -
        0.01958 / oc1_72 -
        0.2876 * np.log(silt) -
        0.0709 * np.log(oc1_72) -
        44.6 * np.log(bd) -
        0.02264 * bd * clay +
        0.0896 * bd * oc1_72 +
        0.00718 * clay * t
    ) + 1
    
    return n
from math import pow

def equation_wc( theta_R, theta_s, alpha, n, mbar):
    m = 1 - (1 / n)
    a=pow(alpha*mbar, n)
    r=theta_R + (theta_s - theta_R) / (pow(1 + a, m))
    return r


def calc_M_i( WC_i, WC_fc, depth):
    # To convert from water content to soil moisture deficit (mm) used by RothC the following equation is used
    return (WC_i - WC_fc) * 10 * depth

def calc_Mis(
    silt,
    clay,
    bulk_density,
    oc,
    t: float = 1.0,
    theta_R: float = 0.01   ,
    soil_thickness: float = 23,
):
    # From Annex A Pedotransfer functions
    # used to calculate the hydraulic properties
    # At Farina et al 2013

    theta_s = calculate_theta_s(silt, clay, oc, bulk_density, t)
    alpha = calculate_alpha(silt, clay, oc, bulk_density, t)
    n = calculate_n(silt, clay, oc, bulk_density, t)

    # Water content at field capacity
    WCfc =     equation_wc(theta_R, theta_s, alpha, n, calc_mbar(-0.05))
    WCb =     equation_wc(theta_R, theta_s, alpha, n, calc_mbar(-1))
    # pwp - permanent wilting point
    WCpwp =     equation_wc(theta_R, theta_s, alpha, n, calc_mbar(-15))
    # capillary water retained at − 1000 bar
    WCc =     equation_wc(theta_R, theta_s, alpha, n, calc_mbar(-1000))

    Mb = calc_M_i(WCb, WCfc, soil_thickness)
    Mpwp = calc_M_i(WCpwp, WCfc, soil_thickness)
    Mc = calc_M_i(WCc, WCfc, soil_thickness)
    
    return Mb, Mpwp, Mc
