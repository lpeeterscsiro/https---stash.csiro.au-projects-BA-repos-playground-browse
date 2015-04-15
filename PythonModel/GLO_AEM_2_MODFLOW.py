#  Interpolate AEM model outcomes to Avon and Karuah
"""

"""
import numpy as np
import os as os
import sys
from scipy.interpolate import griddata, interp1d

#%% Set working directory and load data
runID = ''
if len(sys.argv) != 1:
  runID = '_' + sys.argv[1]

#wdir        = r'C:\Users\pee035\Documents\Projects\BA\Gloucester\GLO_AEM\repo' + runID
#wdir        = r'D:\\BA\\PythonModel' + runID
wdir        = r'C:\\BAModelapp\\PythonModel'+ runID

os.chdir( wdir )
fnames = ['GLO_Avon_baseline.chd',
          'GLO_Avon_crdp.chd',
          'GLO_Karuah_baseline.chd',
          'GLO_Karuah_crdp.chd']
# check if AEM run was succesfull
with open('GLO_AEM_h_gwtargets_baseline.csv') as f:
    test = f.readline()
if 'Failed' in test:
    for fname in fnames:
        np.savetxt(fname,np.ones(1),header='Model Run Failed')
else:
    # MODFLOW info
    dh_chd = np.loadtxt('GLO_AEM_parameters.csv',delimiter=',',skiprows=1, usecols=(1,))[15]
    A_ibound = np.loadtxt( 'GLO_Avon_IBOUND.asc',skiprows=6 )
    A_coords = np.genfromtxt( 'GLO_Avon_IBOUND.asc', usecols=(1) )[0:6]
    K_ibound = np.loadtxt( 'GLO_Karuah_IBOUND.asc',skiprows=6 )
    K_coords = np.genfromtxt( 'GLO_Karuah_IBOUND.asc', usecols=(1) )[0:6]
    A_href = np.loadtxt('GLO_Avon_href.dat') - dh_chd
    K_href = np.loadtxt('GLO_Karuah_href.dat') - dh_chd
    mf_sp = 365.25/12 # modflow stress period length
    mf_nsp = 120*12 # number of stress periods
    mf_ts = np.arange(0,mf_nsp*mf_sp,mf_sp)
    # AEM output
    AEM_h1 = np.loadtxt( 'GLO_AEM_h_gwtargets_baseline.csv',delimiter=',',skiprows=1)
    AEM_h2 = np.loadtxt( 'GLO_AEM_h_gwreceptors_baseline.csv',delimiter=',',skiprows=1)[:,1::]
    AEM_h_b = np.concatenate((AEM_h1,AEM_h2),axis=0)
    AEM_h1 = np.loadtxt( 'GLO_AEM_h_gwtargets_crdp.csv',delimiter=',',skiprows=1)
    AEM_h2 = np.loadtxt( 'GLO_AEM_h_gwreceptors_crdp.csv',delimiter=',',skiprows=1)[:,1::]
    AEM_h_c = np.concatenate((AEM_h1,AEM_h2),axis=0)
    AEM_tsteps = AEM_h_b.shape[1]-2
    AEM_time   = 12*365.25 + 365.25*(np.cumsum(np.concatenate((np.ones(27),np.ones(16)*5)))) #start at 1995, 12 years after 1983
    #%% Create coordinates and indices of active cells
    A_x = np.arange(A_coords[2]+A_coords[4]/2.0,A_coords[2]+(A_coords[0]*A_coords[4])+A_coords[4]/2.0,A_coords[4])
    A_y = np.arange(A_coords[3]+(A_coords[1]*A_coords[4])+A_coords[4]/2.0,A_coords[3],-A_coords[4])
    A_xx,A_yy = np.meshgrid(A_x,A_y)
    A_i,A_j = np.where(A_ibound==1)
    K_x = np.arange(K_coords[2]+K_coords[4]/2.0,K_coords[2]+(K_coords[0]*K_coords[4])+K_coords[4]/2.0,K_coords[4])
    K_y = np.arange(K_coords[3]+(K_coords[1]*K_coords[4])+K_coords[4]/2.0,K_coords[3],-K_coords[4])
    K_xx,K_yy = np.meshgrid(K_x,K_y)
    K_i,K_j = np.where(K_ibound==1)
    #%% Interpolate spatially
    A_h_b = np.zeros((len(A_i),AEM_tsteps))
    K_h_b = np.zeros((len(K_i),AEM_tsteps))
    A_h_c = np.zeros((len(A_i),AEM_tsteps))
    K_h_c = np.zeros((len(K_i),AEM_tsteps))
    for i in range(AEM_tsteps):
        A_h_b[:,i] = griddata(AEM_h_b[:,0:2],
                              AEM_h_b[:,i+2],
                              np.array([A_xx[A_i,A_j],A_yy[A_i,A_j]]).T,
                              method = 'linear',
                              fill_value = 0.0 )
        K_h_b[:,i] = griddata(AEM_h_b[:,0:2],
                              AEM_h_b[:,i+2],
                              np.array([K_xx[K_i,K_j],K_yy[K_i,K_j]]).T,
                              method = 'linear',
                              fill_value = 0.0 )
        A_h_c[:,i] = griddata(AEM_h_c[:,0:2],
                              AEM_h_c[:,i+2],
                              np.array([A_xx[A_i,A_j],A_yy[A_i,A_j]]).T,
                              method = 'linear',
                              fill_value = 0.0 )
        K_h_c[:,i] = griddata(AEM_h_c[:,0:2],
                              AEM_h_c[:,i+2],
                              np.array([K_xx[K_i,K_j],K_yy[K_i,K_j]]).T,
                              method = 'linear',
                              fill_value = 0.0 )
    #%% Interpolate temporally
    A_chd_b = np.zeros((len(A_i),mf_nsp))
    A_chd_c = np.zeros((len(A_i),mf_nsp))
    for i in range(len(A_i)):
        f = interp1d( AEM_time, A_h_b[i,:], kind='slinear', bounds_error=False, fill_value=0.0 )
        A_chd_b[i,:] = A_href[A_i[i],A_j[i]] + f(mf_ts)
        f = interp1d( AEM_time, A_h_c[i,:], kind='slinear', bounds_error=False, fill_value=0.0 )
        A_chd_c[i,:] = A_href[A_i[i],A_j[i]] + f(mf_ts)
    
    K_chd_b = np.zeros((len(K_i),mf_nsp))
    K_chd_c = np.zeros((len(K_i),mf_nsp))
    for i in range(len(K_i)):
        f = interp1d( AEM_time, K_h_b[i,:], kind='slinear', bounds_error=False, fill_value=0.0 )
        K_chd_b[i,:] = K_href[K_i[i],K_j[i]] + f(mf_ts)
        f = interp1d( AEM_time, K_h_c[i,:], kind='slinear', bounds_error=False, fill_value=0.0 )
        K_chd_c[i,:] = K_href[K_i[i],K_j[i]] + f(mf_ts)
    
    #%% Write CHD file
    A_layer = 2
    A_rows  = A_i + 1
    A_columns = A_j + 1
    f_A_chd = open(fnames[0],'w')
    f_A_chd.write('# Change in head in weathered zone from AEM model: baseline\n')
    f_A_chd.write('# Generated with GLO_AEM_2_MODFLOW\n')
    f_A_chd.write('%i # MXACTC\n' % len(A_i))
    for i in range(mf_nsp):
        f_A_chd.write('%i 0\n' % len(A_i))
        for j in range(len(A_i)):
            f_A_chd.write('%i %i %i %10.5e %10.5e\n' % (A_layer, A_rows[j], A_columns[j], A_chd_b[j,i], A_chd_b[j,i]))
    
    f_A_chd.close()
    
    f_A_chd = open(fnames[1],'w')
    f_A_chd.write('# Change in head in weathered zone from AEM model: CRDP\n')
    f_A_chd.write('# Generated with GLO_AEM_2_MODFLOW\n')
    f_A_chd.write('%i # MXACTC\n' % len(A_i))
    for i in range(mf_nsp):
        f_A_chd.write('%i 0\n' % len(A_i))
        for j in range(len(A_i)):
            f_A_chd.write('%i %i %i %10.5e %10.5e\n' % (A_layer, A_rows[j], A_columns[j], A_chd_c[j,i], A_chd_c[j,i]))
    
    f_A_chd.close()
    
    #%% Write CHD file
    K_layer = 2
    K_rows  = K_i + 1
    K_columns = K_j + 1
    f_K_chd = open(fnames[2],'w')
    f_K_chd.write('# Change in head in weathered zone from AEM model: baseline\n')
    f_K_chd.write('# Generated with GLO_AEM_2_MDOFLOW\n')
    f_K_chd.write('%i # MXACTC\n' % len(K_i))
    for i in range(mf_nsp):
        f_K_chd.write('%i 0\n' % len(K_i))
        for j in range(len(K_i)):
            f_K_chd.write('%i %i %i %10.5e %10.5e\n' % (K_layer, K_rows[j], K_columns[j], K_chd_b[j,i], K_chd_b[j,i]))
    
    f_K_chd.close()
    
    f_K_chd = open(fnames[3],'w')
    f_K_chd.write('# Change in head in weathered zone from AEM model: baseline\n')
    f_K_chd.write('# Generated with GLO_AEM_2_MDOFLOW\n')
    f_K_chd.write('%i # MXACTC\n' % len(K_i))
    for i in range(mf_nsp):
        f_K_chd.write('%i 0\n' % len(K_i))
        for j in range(len(K_i)):
            f_K_chd.write('%i %i %i %10.5e %10.5e\n' % (K_layer, K_rows[j], K_columns[j], K_chd_c[j,i], K_chd_c[j,i]))
    
    f_K_chd.close()
