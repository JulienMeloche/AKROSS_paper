'''

Includes functions used in multiple notebooks

'''

import os
import numpy as np
from math import cos, asin, sqrt

# Need forked snowmicropyn from https://github.com/mjsandells/snowmicropyn
from snowmicropyn import density_ssa, profile

from smrt import make_ice_column, make_interface, make_snowpack
from smrt.core.globalconstants import PSU
from smrt.interface.geometrical_optics_backscatter import GeometricalOpticsBackscatter
from smrt.interface.iem_fung92_brogioni10 import IEM_Fung92_Briogoni10


#turn off warning from smrt
import warnings
warnings.simplefilter('ignore')


#constant Ice parameter
MYI_ice = make_ice_column(ice_type='firstyear',
                    thickness=[2], temperature=260, 
                    microstructure_model='independent_sphere',
                    radius=1e-3,
                    brine_inclusion_shape='spheres',
                    density=910,
                    salinity=5*PSU,
                    add_water_substrate=True)

FYI_ice = make_ice_column(ice_type='multiyear',
                    thickness=[3], temperature=260, microstructure_model='independent_sphere',
                    radius=1e-3,
                    brine_inclusion_shape='spheres',
                    density=880,
                    salinity=5*PSU,
                    add_water_substrate=True)


def find_site_smp_files(site, data_dir=None):
    '''
    Finds all smp files for a particular site (subfolder)
    
    e.g. find_site_smp_files('Eureka_6_FYI', data_dir)
    
    '''
    if data_dir is None:
        data_dir = '../DATA/SMP/Sites'
        
    # Get list of filenames for that site
    smp_filelist = os.listdir(path=data_dir + '/' + site)
    # Amend filelist to include path
    return [data_dir + '/' + site + '/' + f for f in smp_filelist], [site[:-4] for f in smp_filelist]


def get_coords(list_of_sites, data_dir):
    '''
    Returns list of lat / long coordinates for list of sites
    
    '''
    
    all_sites = [find_site_smp_files(s, data_dir) for s in list_of_sites]
    # Flatten list of lists
    all_sites = [item for sublist in all_sites for item in sublist]
    # Filter non .pnt items
    all_sites = [i for i in all_sites if i.lower().endswith(".pnt")]

    # Extract coords from all SMP files
    coords = []
    [coords.append(profile.Profile(s).coordinates) for s in all_sites]

    lats = [s[0] for s in coords]
    lons = [s[1] for s in coords]
    return lats, lons



def debye_calc(ssa, density):
    return 4 * (1 - density / 917) / (ssa * 917)

def build_snowpack_pickle(smp_profile, ice_salinity = 5, ice_temp = 260, sea_ice_density = 910, 
                            ice_type = 'firstyear', sigma_surface = 0):
    #build snowpack for CB
    list_sp = []
    for profile in smp_profile:
        ice = make_ice_column(ice_type=ice_type,
                thickness=[2], temperature= ice_temp, 
                microstructure_model='independent_sphere',
                radius=1e-3,
                brine_inclusion_shape='spheres',
                density=sea_ice_density,
                salinity=ice_salinity*PSU,
                add_water_substrate=True)

        # Convert to exp. corr length
        debye = 1.2
        lex_array = debye * 4 * (1 - profile.density / 917) / (profile.ssa * 917)

        profile.temperature[profile.temperature < 251] = 251
        snowpack = make_snowpack(profile.thick, microstructure_model='unified_scaled_exponential',
                        density=profile.density , 
                        porod_length=lex_array,
                        temperature = profile.temperature,
                        salinity=profile.salinity * PSU,
                        polydispersity = 1) + ice
        snowpack.sigma_surface = sigma_surface
        list_sp.append(snowpack)
    
    return list_sp



def change_roughness(list_of_snowpack, mean_ice_rms, mean_ice_lc, mean_snow_rms, mean_snow_lc):
    for i in range(len(list_of_snowpack)):
        #ice
        list_of_snowpack[i].interfaces[-1] = make_interface(IEM_Fung92_Briogoni10, roughness_rms = mean_ice_rms, corr_length = mean_ice_lc)
        #surface
        list_of_snowpack[i].interfaces[0] = make_interface(IEM_Fung92_Briogoni10, roughness_rms = mean_snow_rms, corr_length = mean_snow_lc)
    


def change_roughness_geo(list_of_snowpack, mean_ice_rms, mean_ice_lc, mean_snow_rms, mean_snow_lc):
    for i in range(len(list_of_snowpack)):

        #ice
        ice_mss = 2*mean_ice_rms**2/mean_ice_lc**2
        list_of_snowpack[i].interfaces[-1] = make_interface(GeometricalOpticsBackscatter, mean_square_slope = ice_mss)
        #surface
        snow_mss = 2*mean_snow_rms**2/mean_snow_lc**2
        list_of_snowpack[i].interfaces[0] = make_interface(GeometricalOpticsBackscatter, mean_square_slope = snow_mss)

        


def align_waveform_with_sim(obs_wave):
    #index that correspond to 50% max power in simulation
    gate_sim = 164
    #find gate in obs that correspond to 50% power
    max_sig_obs = np.max(obs_wave)
    for idx, sig in enumerate(obs_wave):
        if (max_sig_obs*0.5 - sig) <=0:
            gate_obs = idx
            break

    gate_diff = gate_obs - gate_sim
    #fill end of waveform with last value
    fill_obs = np.ones(np.abs(gate_diff)) * obs_wave[-1]

    if gate_diff >= 0:
        #fill end of waveform with last value
        fill_obs = np.ones(np.abs(gate_diff)) * obs_wave[-1]
        # remove beginning to match gate of 50% power and add fill value at the end to keep size intact
        new_obs = np.concatenate((obs_wave[gate_diff:], fill_obs)) 
    else:
        #fill end of waveform with last value
        fill_obs = np.ones(np.abs(gate_diff)) * obs_wave[1]
        # remove beginning to match gate of 50% power and add fill value at the end to keep size intact
        new_obs = np.concatenate((fill_obs, obs_wave[:gate_diff])) 
        
    
    return new_obs

def calc_A(backscatter):
    # Eqn 18
    # Should be mean backscatter at a single site

    return np.sqrt(np.mean(backscatter ** 4) / np.mean(backscatter **2))

def normalization_factor(sim, obs):
    # Eqn 17
    return (np.sum(obs*sim) - np.sum(obs)*np.sum(sim)) / (np.sum(obs**2) - (np.sum(obs))**2)



def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(hav))

def closest(data, v):
    #return index to find waveform
    pos = min(data, key=lambda p: distance(v['lat'],v['lon'],p['lat'],p['lon']))
    print(f'return index for position : {pos}')
    
    return data.index(min(data, key=lambda p: distance(v['lat'],v['lon'],p['lat'],p['lon'])))



















# def smp_snowpacks(list_of_filenames, list_of_file_sites, salt_dict, temp_dict, layer_thickness=0.05, rough=False, ice_lens=None, permittivity=None, 
#                   sea_ice_density=910, sea_ice_temp = 260, ice_salinity = 5, sigma_surface = 0):
#     # Function to take in an SMP measurement file, derive density and SSA
#     # And generate SMRT snowpack
#     # layer_thickness governs the thickness of layers to be used in SMRT
#     # default layer_thickness is 5cm    
        
#     snowpacks = []
    
#     # King 2020 coefficients. SSA are from TVC and assumed to apply to Eureka / Alert
#     #k2020_coeffs = {'density':[312.54, 50.27, -50.26, -85.35], 'ssa':[12.05, -12.28, -1], 'equation':'ssa'}
#     k2020_coeffs = {'density':[312.54, 50.27, -50.26, -85.35], 'ssa':[2.37, -0.7, -0.06], 'equation':'ssa'}
#     #k2020_coeffs = {'density':[312.54, 50.27, -50.26, -85.35], 'ssa':[9.212, -14.627, -1.582], 'equation':'ssa'}
    
#     # Loop over SMP files for site:
#     for smp, site in zip(list_of_filenames, list_of_file_sites):
        
        
#         # 0. Determine substrate
#         if 'FYI' in smp:
#             ice = make_ice_column(ice_type='firstyear',
#                     thickness=[2], temperature=sea_ice_temp, 
#                     microstructure_model='independent_sphere',
#                     radius=1e-3,
#                     brine_inclusion_shape='spheres',
#                     density=sea_ice_density,
#                     salinity=ice_salinity*PSU,
#                     add_water_substrate=True)
#         else:
#             ice = make_ice_column(ice_type='multiyear',
#                     thickness=[3], temperature=sea_ice_temp, microstructure_model='independent_sphere',
#                     radius=1e-3,
#                     brine_inclusion_shape='spheres',
#                     density=sea_ice_density,
#                     #brine_permittivity_model = seawater_permittivity_stogryn95,            
#                     salinity=ice_salinity*PSU,
#                     add_water_substrate=True)
        
        
#         # 1. Take median and process density / SSA
#         #smp_median = density_ssa.median_profile([smp])
#         try:
#             smp_profile = profile.Profile(smp)
#             c2020_m = density_ssa.calc(smp_profile.samples_within_snowpack(), coeff_model=k2020_coeffs, window=5, overlap=50)
#             total_depth_in_m = c2020_m.distance.iloc[-1] * 1e-3

#             # 2. Group density / SSA into layers. Take median
#             current_thickness = c2020_m.distance.diff().iloc[-1] * 1e-3 # Convert to m
#             number_in_group = int(layer_thickness / current_thickness)
#             df = c2020_m.rolling(number_in_group).median()
#             df = df.iloc[number_in_group::number_in_group, :]

#             # 3. Calculate number of layers needed
#             nlayers = len(df)

#             # 4. Assign layer thickness from top, with any remaining depth allocated to bottom layer
#             thickness_array = nlayers * [layer_thickness]
#             # Add in remaining profile to lowest layer
#             extra_thickness = total_depth_in_m - sum(thickness_array)
#             thickness_array[-1] += extra_thickness
            
#             # 5. Extract density
#             density_array = df.density

#             # 6. Derive exp corr length
#             ssa_array = np.exp(df.ssa)
#             # Convert to exp. corr length
#             debye = 1
#             lex_array = debye * 4 * (1 - density_array / 917) / (ssa_array * 917)

                
#             # Convert to list once lex has been calculated
#             # This enables ice lens to be inserted
#             lex_array = lex_array.tolist()
#             density_array = density_array.tolist()    


#             #get height from salt and temp interpolation
#             height = np.cumsum(thickness_array[::-1])

#             #get poly
#             poly_ws = 0.7*np.ones(int(len(height)/2))
#             poly_dh = 1.3*np.ones(int(len(height)/2)+ int(len(height)%2))
#             poly = np.concatenate([poly_ws, poly_dh])

#             #set salinity of last snow layer
#             salt_pit = salt_dict[site]
#             snow_salinity = np.interp(height, salt_pit.height[::-1].values/100, salt_pit.param[::-1])[::-1]

#             #settemp array
#             temp_pit = temp_dict[site]
#             temp_pit.param[temp_pit.param <-22] = -22
#             temperature_array = np.interp(height, temp_pit.height[::-1].values/100, temp_pit.param[::-1])[::-1] + 273

                
#             # 8. Insert ice lens if needed
#             # Have to insert it when creating the snowpack so boundaries (interfaces) are calculated correctly
#             if isinstance(ice_lens, int) and (abs(ice_lens) <= len(thickness_array)):
#                 # Will only insert ice lens if an integer is passed
#                 thickness_array = np.insert(thickness_array,ice_lens+1, 0.002) # 2mm ice lens below layer specifed
#                 lex_array = np.insert(lex_array, ice_lens+1, 1e-5) # 0.1mm exp correlation length
#                 density_array = np.insert(density_array, ice_lens+1, 909) # Watts et al., TC 2016.
#                 temperature_array = np.insert(temperature_array, ice_lens+1, 260)
#             elif ice_lens is None:
#                 pass
#             else:
#                 print ('ice_lens should be integer equal to or less than the number of snow layers')
#                 return

#             # 9. Generate snowpack
#             if rough == False:
#                 snow_ice = make_snowpack(thickness_array, microstructure_model='unified_scaled_exponential',
#                                         ice_permittivity_model=permittivity, density=density_array, 
#                                         porod_length=lex_array, temperature=temperature_array,
#                                         salinity = snow_salinity*PSU, polydispersity = poly) + ice
#                 snow_ice.sigma_surface = sigma_surface
#                 snowpacks.append(snow_ice)

#             else:
#                 surface = make_interface(GeometricalOpticsBackscatter, mean_square_slope=0.05)
#                 snow_ice = make_snowpack(thickness_array, microstructure_model='unified_scaled_exponential', surface=surface,
#                                 ice_permittivity_model=permittivity, density=density_array, 
#                                 porod_length=lex_array, temperature=temperature_array,
#                                 salinity = snow_salinity*PSU, polydispersity = poly) + ice
#                 snow_ice.sigma_surface = sigma_surface
#                 snowpacks.append(snow_ice)
#         except:
#             print ('make snowpack failed for', smp)
#             pass
            
#     return snowpacks