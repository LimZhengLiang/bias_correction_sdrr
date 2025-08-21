#Check SDRR max and extrapolation:
import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import netCDF4 as nc                       # netcdf package for reading netcdf data
import pandas as pd                        # pandas package for reading CSV files
from itertools import product
import os
from scipy.ndimage import gaussian_filter

#Smoothing is done to reduce the intensity of green region


#As an alternative, we can use K-means to cluster the different weather systems based on intensity and size. Get the climatological mappings of the various
#features.



def nan_gaussian_filter(data, sigma):
    """
    Apply Gaussian filter to data with NaNs preserved.

    Parameters:
        data (ndarray): Input 2D array with NaNs.
        sigma (float or sequence): Standard deviation for Gaussian kernel.

    Returns:
        ndarray: Smoothed array with NaNs preserved.
    """
    nan_mask = np.isnan(data)
    
    data_zeroed = np.where(nan_mask, 0, data)
    smoothed_data = gaussian_filter(data_zeroed, sigma=sigma)

    # FIXED: invert before converting to float
    valid_mask = (~nan_mask).astype(float)
    norm = gaussian_filter(valid_mask, sigma=sigma)

    with np.errstate(invalid='ignore', divide='ignore'):
        smoothed = smoothed_data / norm
        smoothed[norm == 0] = np.nan

    return smoothed



def create_netcdf( timeobject, latitude, longitude, values, output_filename ): 
    
    '''
    timeobject      - A datetime object
    latitude        - numpy array of latitudes
    longitude       - numpy array of longitudes 
    values          - numpy array of reflectivity values 
    output_filename - our output filename (string)
    '''
    
    # Opening our output filename
    
    ncfile = nc.Dataset( output_filename, 'w' )
    
    # Create dimension 
    
    timed = ncfile.createDimension('Time',None)                # time dimension
    
    latd = ncfile.createDimension('Latitude',len(latitude))    # latitude dimension
    
    lond = ncfile.createDimension('Longitude',len(longitude))  # longitude dimension 
    
    # Create dimension variable
    
    times = ncfile.createVariable('Time','f4',('Time',))       # dimension variable is 32-bit floating point 
    
    latitudes = ncfile.createVariable('Latitude','f4',('Latitude',)) # dimension variable is 32-bit floating point 
    
    longitudes = ncfile.createVariable('Longitude','f4',('Longitude',)) 
    
    # Create field variable (reflectivity) corresponding to dimension
    
    Z = ncfile.createVariable('Zradar','f4',('Time','Latitude','Longitude'),fill_value=-999.,zlib=True)

    # Creating the time dimension
    
    dt = timeobject - datetime(1970,1,1,0,0)
    
    t = dt.total_seconds()
    
    # Assigning dimension variables 
    
    times[:] = t
    
    latitudes[:] = latitude
    
    longitudes[:] = longitude
    
    Z[0,:,:] = values
    
    # Adding attributes 
    
    latitudes.units = 'degrees_north'
    
    longitudes.units = 'degrees_east'
    
    times.units = 'seconds since 1970-01-01 00:00:00'
    
    Z.units = 'dBZ'
    
    Z.long_name = 'Reflectivity'
    
    ncfile.close()


# Example: 2D radar frame (replace with your actual data)
def extract_netcdf2( fil, lat_name, lon_name, var_name ): 

  '''
  fil is a string, name of our netcdf file
  lat_name is the name of our latitude inside netcdf file
  lon_name is the name of our longitude inside netcdf file
  var_name is the name of our variable inside netcdf file

  return latitude, longitude, and variable as numpy arrays   

  '''

  z = nc.Dataset( fil ) 

  lat = z[lat_name][:]

  lon = z[lon_name][:]

  val = z[var_name][:]   # Assuming that val is 3d variable 

  return lat, lon, val   

# define function to read time from wrfout data

def extract_netcdf( fil, lat_name, lon_name, var_name ): 

  '''
  fil is a string, name of our netcdf file
  lat_name is the name of our latitude inside netcdf file
  lon_name is the name of our longitude inside netcdf file
  var_name is the name of our variable inside netcdf file

  return latitude, longitude, and variable as numpy arrays   

  '''

  z = nc.Dataset( fil ) 

  lat = z[lat_name][:]

  lon = z[lon_name][:]

  val = z[var_name][:][0]   # Assuming that val is 3d variable 

  return lat, lon, val   

# fil is wrfout_d03_2022-08-06_11_30_00.nc

def read_time (fil): 
    
  yyyy = fil[10:14]
  
  mm = fil[15:17]
  
  dd = fil[18:20]

  hh = fil[21:23]
  
  mn = fil[24:26]

  return yyyy,mm,dd,hh,mn

import sys   # python package to command line argument 

wea_event = sys.argv[1]

swirls_dir = f"/mnt/c/Users/LZL/Downloads/check_binning_bias_correction/cmax/{wea_event}/netcdf/" #cmax
nwp_re_dir = f"/mnt/c/Users/LZL/Downloads/check_binning_bias_correction/sdrr_regridded/{wea_event}/" #regridded sdrr
nwpdir = f"/mnt/c/Users/LZL/Downloads/check_binning_bias_correction/sdrr_FIR_original/{wea_event}/" #original sdrr


# Loop through all .nc files in nwpdir
for radar in os.listdir(nwpdir):
    if radar.endswith(".nc"):
        # Full path to NWP file
        #nwpfil = os.path.join(nwpdir, radar)

        # Corresponding radar file (assumes same name exists in cmax dir)
        #nwp_re_fil = os.path.join(nwp_re_dir, radar)
        #swirlsfil = os.path.join(swirls_dir, radar)

        print("Processing:", radar)
        print("  SDRR Original file:     ", nwpdir + radar)
        print("  SDRR Regridded file:     ", nwp_re_dir + radar)
        print("  CMAX file:  ", swirls_dir + radar)

        # --- your processing logic here ---


        # declare netcdf variables of swirls netcdf data files 

        lat_nwp = 'Latitude'

        lon_nwp = 'Longitude'

        var_nwp = 'Zradar'

        lat_nwp_re = 'lat'

        lon_nwp_re = 'lon'

        var_nwp_re = 'Zradar'

        lat_swirls = 'lat'

        lon_swirls = 'lon'

        var_swirls = 'Band1'

        lats_re, lons_re, valnn_re = extract_netcdf( nwp_re_dir + radar, lat_nwp_re, lon_nwp_re, var_nwp_re ) #regridded sdrr
        lats_re, lons_re, vals = extract_netcdf2( swirls_dir + radar, lat_swirls, lon_swirls, var_swirls ) #cmax
 

        lats, lons, valnn = extract_netcdf( nwpdir + radar, lat_nwp, lon_nwp, var_nwp ) #original sdrr
        
        print("CMAX shape:", vals.shape, "Regridded SDRR shape:", valnn_re.shape, "SDRR original shape:", valnn.shape)

        #Obtain csv data based on classification
        valnn_re = valnn_re.data if hasattr(valnn_re, 'data') else valnn_re

        vals = vals.data.copy()
        vals[vals == -10000] = -25
        #valn[valn <= 0] = -25
        vals = np.nan_to_num(vals, nan=-25)
        
        
        
        filename="quantile_comparison_20250206_to_20250527_neg25_percentile_fnlv2.csv"


        csv_file = f"/mnt/c/Users/LZL/Downloads/clima_bin_fnl_csv/{filename}"
        def read_percentiles_from_csv(csv_file, cmax_col='CMAX_dBZ', sdrr_col='SDRR_dBZ'):
        #def read_percentiles_from_csv(csv_file, cmax_col='Band1', sdrr_col='Zradar'):    
            '''
            Read percentile data from CSV file
            
            csv_file: path to CSV file
            cmax_col: column name for CMAX data (swirls percentiles)
            sdrr_col: column name for SDRR data (nwp percentiles)
            
            Returns: swirlsp, nwpp (percentile arrays)
            '''
            
            df = pd.read_csv(csv_file)
            
            # Extract percentile values from specified columns
            swirlsp = df[cmax_col].values  # CMAX percentiles
            nwpp = df[sdrr_col].values     # SDRR percentiles
            #perc = df["Percentile"].values
            return swirlsp, nwpp, df


        #valnn = np.full((2, 2), 48)

        nwp_corrected = np.zeros( shape = (valnn.shape) )
        print("Reading percentiles from CSV file...")
        swirlsp, nwpp, df = read_percentiles_from_csv(csv_file)




        def smooth_quantile_mapping(valnn, nwpp, swirlsp, slope_fraction=0.5):
            """
            Linearly interpolate within range using bin midpoints,
            and smoothly extrapolate above and below using gentler slopes.

            Parameters:
                valnn: ndarray - Input values to correct.
                nwpp: ndarray - NWP quantile levels (any order).
                swirlsp: ndarray - Swirls quantile levels (same size as nwpp).
                slope_fraction: float - Slope reduction factor for extrapolation.

            Returns:
                interp_valnn: ndarray - Bias-corrected values,
                            capped so it never exceeds the original valnn.
            """
            # Sort both arrays to ascending order
            idx_sort = np.argsort(nwpp)
            x = nwpp[idx_sort]
            y = swirlsp[idx_sort]

            interp_valnn = np.full_like(valnn, np.nan, dtype=np.float32)

            # Bin midpoint method
            midpoint_valnn = np.full_like(valnn, np.nan, dtype=np.float32)
            for i in range(len(x) - 1):
                idx_bin = (valnn > x[i]) & (valnn <= x[i+1])
                midpoint_valnn[idx_bin] = 0.5 * (y[i] + y[i+1])

            # Linear interpolation
            interp_possible = np.isfinite(x).all() and np.isfinite(y).all()
            if interp_possible:
                linear_valnn = np.interp(valnn, x, y, left=np.nan, right=np.nan)
            else:
                linear_valnn = np.full_like(valnn, np.nan, dtype=np.float32)

            # Default: combine midpoint & linear
            interp_valnn = np.fmax(midpoint_valnn, linear_valnn)

            # Special case: values between last two quantiles - use interpolation only
            mask_last_interval = (valnn > x[-2]) & (valnn <= x[-1])
            interp_valnn[mask_last_interval] = linear_valnn[mask_last_interval]
            print(x[-1], x[-2], y[-1], y[-2])
            # Extrapolate above max
            mask_high = valnn > x[-1]
            if np.any(mask_high):
                slope_high = (y[-1] - y[-2]) / (x[-1] - x[-2]) * slope_fraction
                interp_valnn[mask_high] = y[-1] + slope_high * (valnn[mask_high] - x[-1])

            # Safeguard: do not allow corrected values > original valnn
            interp_valnn = np.minimum(interp_valnn, valnn)

            return interp_valnn


        # Step 1: Remove nonphysical values before smoothing
        interpolated_valnn = smooth_quantile_mapping(valnn, nwpp, swirlsp, slope_fraction=0.5)
        #valnn[valnn <= 0] = np.nan

        # Step 2: Apply NaN-aware Gaussian smoothing
        #smoothed_valnn = nan_gaussian_filter(valnn, sigma=1)

        # Step 3: After smoothing, keep only values >13 and not NaN
        #val_new = np.where((~np.isnan(smoothed_valnn)), smoothed_valnn, np.nan)

           
        #nwp_corrected = (0.8 * val_new) + (0.2 * interpolated_valnn)  
        #formula1 = (0.9 * valnn) + (0.1 * interpolated_valnn)
        #formula2 = (0.85 * valnn) + (0.15 * interpolated_valnn) 
        #formula3 = (0.8 * valnn) + (0.2 * interpolated_valnn)   
        #formula4 = (0.75 * valnn) + (0.25 * interpolated_valnn)
        #formula5 = (0.7 * valnn) + (0.3 * interpolated_valnn)
        #nwp_corrected = np.maximum.reduce([formula1, formula2, formula3, formula4, formula5])                     
        
        
        
        
        nwp_corrected = interpolated_valnn


        # Clip to maximum of valnn (ignoring NaNs if present)
        #valnn_max = np.nanmax(valnn)  
        #print("Valn Max::", valnn_max)
        #nwp_corrected = np.clip(nwp_corrected2, None, valnn_max)
        #nwp_corrected = intensity_weighted_gaussian_filter(nwp_corrected, sigma=2.5)   
        #nwp_corrected = nan_gaussian_filter(nwp_corrected, sigma=1)  
        #nwp_corrected[nwp_corrected <= 5] = np.nan   
        print("interpolated valnn shape", interpolated_valnn.shape)
        
        print("nwp corrected shape", nwp_corrected.shape)
        #print("Corrected value:", nwp_corrected)

        
        #Create netcdf
        from datetime import datetime  # python package for manipulating time 

        print(radar)

        yyyy, mm, dd, hh, mn = read_time (radar)

        timeobject = datetime(int(yyyy),int(mm),int(dd),int(hh),int(mn),0)
        
        
        output_dir = f"/mnt/c/Users/LZL/Downloads/check_binning_bias_correction/interpolated_valnn/{wea_event}/"
        os.makedirs(output_dir, exist_ok=True)
        template = 'radfr_d03_%Y-%m-%d_%H_%M_00.rapids.nc' # this is how we want our final output to look like 
        output_filename = output_dir + timeobject.strftime(template)

        latitude = lats

        longitude = lons 

        values = nwp_corrected

        create_netcdf( timeobject, latitude, longitude, values, output_filename )

