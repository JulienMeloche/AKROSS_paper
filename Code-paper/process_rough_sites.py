# import roughnessFunction as rough
import pandas as pd
import numpy as np
from plyfile import PlyData
import random
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.stats import binned_statistic
import os

def create_sperical_mask(x, y, z, center, radius):

    dist_from_center = np.sqrt((x - center[0])**2 + (y-center[1])**2 + (z-center[2])**2)
    mask = dist_from_center <= radius
    
    return mask

def SVD(X):
    """
    This method will fit a plane to the data

    X : np.array of point (x,y,z) (like data or subdata)

    Singular value decomposition method.
    Source: https://gist.github.com/lambdalisue/7201028
    """
    # Find the average of points (centroid) along the columns
    C = np.average(X, axis=0)

    # Create CX vector (centroid to point) matrix
    CX = X - C
    # Singular value decomposition
    U, S, V = np.linalg.svd(CX)
    # The last row of V matrix indicate the eigenvectors of
    # smallest eigenvalues (singular values).
    N = V[-1]
    # Extract a, b, c, d coefficients.
    x0, y0, z0 = C
    a, b, c = N
    d = -(a * x0 + b * y0 + c * z0)

    #returns parameters of the fitted plane
    return a, b, c, d

def perp_error(params, x, y, z):
    """
    standard deviation of perpendicular distance (or rmsh) of the
    'xyz' points, to the plane defined by the coefficients 'a,b,c,d' in
    'params'.
    """
    a, b, c, d = params

    length = np.sqrt(a**2 + b**2 + c**2)
    dist = a * x + b * y + c * z + d / length
    rms = np.std(dist)

    return rms, dist

def transformCoord(params, x, y, z):
    """
    This function is needed for Corrlength. This function creates new coordinate (x,y,z)
    from the fitted plane define earlier. The fitted plane becomes the new xy plane. This
    step was needed in my case because the 3d point cloud had arbitrary orientation.
    You can run it anyway and the coordinates should not change much if you surface
    is leveled and that z coord really represent surface height (not my case).
    """

    a, b, c, d = params
    #Definition of the rotation matrix
    norm = np.sqrt(a**2+b**2)
    R = np.array( [[b/norm, -a/norm, 0],
              [a*c/norm, b*c/norm, -norm],
              [a, b, c]] )

    #Calculation of new point in new coordinate system
    x_i, y_i, z_i = [], [], []
    for i in range(len(x)):
        P = np.array([x[i],y[i],z[i]])
        P_temp = R.dot(P)
        x_i.append(P_temp[0])
        y_i.append(P_temp[1])
        z_i.append(P_temp[2])

    new_x = np.array(x_i)
    new_y = np.array(y_i)
    # translation to set point cloud to origin
    new_z = np.array(z_i) + d

    return new_x, new_y, new_z

def acf1d_fit_exp(r, acf1d, r_max):
    """
        fit the correlation data acf1d for given lags r in the range [0,r_max] 
        to an exponential
        returns:
    """
    
    # set fitrange
    fitrange = (r < r_max)
    # define residual function for least squares fit
    def residual( p, r, acf ):
        C0 = p[0]
        correlation_length = p[1]
        return ( C0*np.exp( -r/correlation_length) - acf )

    # initial values for the optimization
    p0 = np.array([1,0.1])

    # least square fit in the required range
    p_opt, info = opt.leastsq(residual, p0, args=(r[fitrange],acf1d[fitrange]))
    C0 = p_opt[0]
    correlation_length = p_opt[1]
    acf1d_exp = residual( p_opt, r, 0 )
    
    return acf1d_exp, [C0, correlation_length]

def CorrLength(x, y, z, nbPoint, radius):
    """
    This function uses from eqn 2 martinez-agirre et al 2019. Because it is computationnaly intensive 
    (Nb points create NbPoint**2 pairwise points). I had to do the same trick with subdata.
    """
    #Create a matrix of the points (x,y,z)
    data = np.c_[x,y,z]

    #**************************************************
    #Select a subset of points from data
    aux = list(data)
    random.shuffle(aux)
    #aux is data but randomly mix
    subData=aux[0:nbPoint]

    subX = np.array([x[0] for x in subData])
    subY = np.array([y[1] for y in subData])
    subZ = np.array([z[2] for z in subData])
    #subr = np.sqrt(subX**2 + subY**2)
    #*************************************************
    
    
    # get pairwise lag and value of height
    list_lag = []
    for i in range(0,len(subX)):
        z0 = subZ[i]**2
        for j in range(0,len(subX)):
            lag = np.sqrt((subX[j] - subX[i])**2 + (subY[j] - subY[i])**2)
            z0 = subZ[i]**2
            z1 = subZ[i]*subZ[j]

            list_lag.append([lag, z1, z0])
            
    # find acf from eqn 2 martinez-agirre et al 2019
    lag = [l[0] for l in list_lag]
    z1 = [l[1] for l in list_lag]
    z0 = [l[2] for l in list_lag]
    value1 = binned_statistic(lag, z1, statistic='sum', bins=20)
    value0 = binned_statistic(lag, z0, statistic='sum', bins=20)
    acf = np.abs(value1[0]/value0[0])
    
    #get r (lag) from bins values
    bins = value1[1][:-1]
    bins_width = value1[1][1] - value1[1][0]
    r = bins + bins_width
    
    acf_fit, param = acf1d_fit_exp(r, acf, radius*1.75)
    lcorr = param[1]
    
    # plot is needed
    plt.scatter(r, acf)
    plt.plot(r, acf_fit)
    plt.xlabel('lag (m)')
    plt.ylabel('ACF')
    
    return lcorr

def process_roughness(path, radius):
    
    list_rough = []
    for file in sorted(os.listdir(path)):
        if '.ply':
            print(file)
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

                #radius is in meter
                mask = create_sperical_mask(x, y, z, center, radius)
                subregion = data[mask]
                
                #build small subset for SVD
                aux = list(subregion)
                random.shuffle(aux)
                subData=aux[0:1000]

                a, b, c, d = SVD(subData)
                sub_x = np.array([n[0] for n in subregion])
                sub_y = np.array([n[1] for n in subregion])
                sub_z = np.array([n[2] for n in subregion])
                #print("Param for Equation of plane  abcd: {:.3f} {:.3f} {:.3f} {:.3f}".format(a, b, c, d))
                # Orthogonal mean distance from roughnessFunction again
                rms, dist = perp_error((a, b, c, d), sub_x, sub_y, sub_z)
                
                #Find number of points
                nb_point = len(dist)
                list_rms.append(rms)
                list_point.append(nb_point)
                #print('number of points = '+ str(nbPoints))
                #find Corr length
                new_x, new_y, new_z = transformCoord((a, b, c, d), sub_x, sub_y, sub_z)
                #print('Coordinates transform')
                lcorr = CorrLength(new_x, new_y, new_z, 1000, radius)
                if lcorr > 1.0:
                    print('failed')
                    continue
                print(lcorr)
                list_corrL.append(lcorr)

                
            rms_dt = np.mean(list_rms)
            lcorr_dt = np.mean(list_corrL)
            nbPoints = np.mean(list_point)
            print('RMS (m) ='+ str(rms_dt))
            print('Correlation length (m) = '+ str(lcorr_dt))
            print('number of points = '+ str(nbPoints))
            
            list_rough.append({'file' : file, 'rms': rms_dt, 'lcorr' : lcorr_dt, 'NbPoints' : nbPoints})

    return list_rough


# this part build data from PLY filename.

path = "/home/jum002/store7/data/AKROSS_data/roughness_pointcloud"


df = pd.DataFrame(process_roughness(path, 0.4))
df.to_csv('/home/jum002/code-workshop/AKROSS_paper/Code-paper/smrt_in-out/rough_param_detrend.csv', index = False)


