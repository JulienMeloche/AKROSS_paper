# import roughnessFunction as rough
import pandas as pd
import numpy as np
from plyfile import PlyData
import random
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.stats import binned_statistic
import os

import process_rough_sites as rough

path = "/home/jum002/store5/data/AKROSS_data/roughness_pointcloud"

list_radius = []
for radius in np.linspace(0.1, 1.5, 10):
    print(f'radius = {radius}')

    list_ice, list_ice_rough, list_snow = [], [], []
    for file in sorted(os.listdir(path)):
        if '.ply':
            points = PlyData.read(os.path.join(path,file))
            #This create an array of x value for all point points
            x = points.elements[0].data['x']
            y = points.elements[0].data['y']
            z = points.elements[0].data['z']
            data = np.c_[x,y,z]
            # Solve using SVD method
            list_rms, list_corrL, list_point = [], [], []
            for m in np.arange(0,20):
                center = data[random.randrange(0, len(x))]
                # radius = 0.3
                #radius is in meter
                mask = rough.create_sperical_mask(x, y, z, center, radius)
                subregion = data[mask]
                
                #build small subset for SVD
                aux = list(subregion)
                random.shuffle(aux)
                subData=aux[0:1000]

                a, b, c, d = rough.SVD(subData)
                sub_x = np.array([n[0] for n in subregion])
                sub_y = np.array([n[1] for n in subregion])
                sub_z = np.array([n[2] for n in subregion])
                #print("Param for Equation of plane  abcd: {:.3f} {:.3f} {:.3f} {:.3f}".format(a, b, c, d))
                # Orthogonal mean distance from roughnessFunction again
                rms, dist = rough.perp_error((a, b, c, d), sub_x, sub_y, sub_z)
                
                #Find number of points
                nb_point = len(dist)
                list_rms.append(rms)
                list_point.append(nb_point)
                # #print('number of points = '+ str(nbPoints))
                # #find Corr length
                # new_x, new_y, new_z = rough.transformCoord((a, b, c, d), sub_x, sub_y, sub_z)
                # #print('Coordinates transform')
                # lcorr = CorrLength(new_x, new_y, new_z, 1000)
                # if lcorr > 1.0:
                #     print('failed')
                #     continue
                # print(lcorr)
                # list_corrL.append(lcorr)

                
            rms_dt = np.mean(list_rms)
            # lcorr_dt = np.mean(list_corrL)
            nbPoints = np.mean(list_point)
            # print('RMS (m) ='+ str(rms_dt))
            # print('Correlation length (m) = '+ str(lcorr_dt))
            # print('number of points = '+ str(nbPoints))
            if ('pressure' in file) or ('rough' in file) or ('AK3_seaice' in file):
                list_ice_rough.append(rms_dt)
            elif 'snow' in file:
                list_snow.append(rms_dt)
            elif 'ice' in file:
                list_ice.append(rms_dt)
            else:
                print(file)
            
    print(f' snow : {np.mean(list_snow)}')
    print(f' ice flat: {np.mean(list_ice)}')
    print(f' ice rough: {np.mean(list_ice_rough)}')
    list_radius.append({'radius' : radius, 
                        'snow' : np.mean(list_snow), 
                        'ice_flat' : np.mean(list_ice), 
                        'ice_rough' : np.mean(list_ice_rough)})

df_radius = pd.DataFrame(list_radius)
df_radius.to_csv('smrt_in-out/roughness_param_scale.csv')