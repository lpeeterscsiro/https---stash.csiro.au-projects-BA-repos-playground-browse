# GLO-AEM: Gloucester AEM model
"""
Simulates drawdown from CSG and coal mines in the weathered zone
at gw receptor locations outside alluvium and drawdown underneath
the alluvium of the Avon and Karuah, to be used as input for the 
alluvium modflow models
This version of the model is streamlined to be run on a cluster
"""
#%% Preamble
import numpy as np
import os as os
import sys
import ttim as ttim

#%% Set working directory, load data and set general model parameters
#runID = ''
#if len(sys.argv) != 1:
#  runID = '_' + sys.argv[1]

wdir        = r'C:\Users\pee035\Documents\Projects\BA\Gloucester\GLO_AEM\PythonModel'
#wdir        = r'D:\\BA\\PythonModel' + runID
#wdir        = r'C:\\BAModelapp\\PythonModel'+ runID

os.chdir( wdir )
# Baseline
mine_p_b  = np.loadtxt('GLO_AEM_mine_footprint_baseline.csv',delimiter=',',usecols=((0,1,2)),skiprows=1)
mine_q_b  = np.loadtxt('GLO_AEM_mine_waterproduction_baseline.csv',delimiter=',',usecols=((0,1,2)),skiprows=1)
nmines_b  = len(np.unique(mine_p_b[:,0]))
# CRDP
wells    = np.loadtxt('GLO_AEM_wellfile.csv',delimiter=',',skiprows=1)
mine_p_c = np.loadtxt('GLO_AEM_mine_footprint_crdp.csv',delimiter=',',usecols=((0,1,2)),skiprows=1)
mine_q_c = np.loadtxt('GLO_AEM_mine_waterproduction_crdp.csv',delimiter=',',usecols=((0,1,2)),skiprows=1)
nwells   = len(wells)
nmines_c = len(np.unique(mine_p_c[:,0]))
# Parameters
params      = np.loadtxt('GLO_AEM_parameters.csv',delimiter=',',skiprows=1, usecols=(1,))
coalseams   = np.loadtxt(r'..\inputs\coalseams\coalseams%04i.csv' % int(params[0]),delimiter=',',skiprows=1)
majorfaults = np.loadtxt(r'..\inputs\majorfaults\majorfaults%04i.csv' % int(params[1]),delimiter=',',skiprows=1)
subfaults   = np.loadtxt(r'..\inputs\subseismicfaults\subseismicfaults%04i.csv' % int(params[2]),delimiter=',',skiprows=1)
boundary    = np.loadtxt('GLO_AEM_BasinBoundary.csv',delimiter=',',skiprows=1)
nseams      = len(coalseams)
nfaults     = len(np.unique(majorfaults[:,0]))
nsubfaults  = len(np.unique(subfaults[:,0]))
# Parameters
params      = np.loadtxt('GLO_AEM_parameters.csv',delimiter=',',skiprows=1, usecols=(1,))
# Output locations and time
gwreceptors = np.loadtxt('GLO_AEM_gwreceptors.csv', delimiter=',', skiprows=1, usecols=(1,2) )
gwID = []
with open('GLO_AEM_gwreceptors.csv') as f:
    f.readline()
    for line in f:
        gwID.append(line.split(',')[0])

gwtargets   = np.loadtxt('GLO_AEM_targets.csv', delimiter=',', skiprows=1)
outtime     = np.cumsum(np.concatenate((np.ones(27),np.ones(16)*5)))
# Output filenames
fnames = ['GLO_AEM_h_gwreceptors_baseline.csv',
          'GLO_AEM_h_gwtargets_baseline.csv',
          'GLO_AEM_h_gwreceptors_crdp.csv',
          'GLO_AEM_h_gwtargets_crdp.csv',
		  'GLO_AEM_CSG_waterproduction.csv']
for fname in fnames:
        np.savetxt(fname,np.ones(1),header='Model Run Failed')
#%% Create layer structure
thick_ws = 75.0 # thickness weatherd zone
thick_cs = 3.0 # thickness of coal seam
naq = nseams+1 # number of aquifers
nll = nseams # number of leaky layers
zaq = np.zeros(2*naq+1)
zaq[0] = 1.0
zaq[2] = -thick_ws # bottom of weathered zone
zaq[3:(2*naq+1):2] = -coalseams[:,1]+0.5*thick_cs # top coal seams
zaq[4:(2*naq+1):2] = -coalseams[:,1]-0.5*thick_cs # bottom coal seams
zmid = zaq[0:-1]+0.5*np.diff(zaq) # position of center of each layer
zmid_aq = zmid[1:len(zaq):2] # center of all aquifers
zmid_ll = zmid[0:len(zaq):2] # center of all leaky layers separating aquifers
zmid_ll[0] = zmid_aq[0]

#%% Hydraulic properties: P=a1*exp(a2*depth)
K_a1 = params[3:5] # a1 for [K interburden, K coal seam ] 
K_a2 = params[5:7] # a2 for [K interburden, K coal seam ]
S_a1 = params[7:9] # a2 for [S interburden, S coal seam ]
S_a2 = params[9:11] # a2 for [S interburden, S coal seam ]
KvKh = params[11] # Kv over Kh for interburden
Kaq = K_a1[1]*np.exp(K_a2[1]*zmid_aq)*365.25 # convert to m/yr
Kaq[0] = K_a1[1]*365.25
Kaq[Kaq<1e-7*365.25] = 1e-7*365.25
Kll = ( K_a1[0]*np.exp(K_a2[0]*zmid_ll)*365.25 )*KvKh # convert to m/yr
Kll[Kll<1e-7*365.25] = 1e-7*365.25    
Saq = S_a1[1]*np.exp(S_a2[1]*zmid_aq)
Saq[0] = S_a1[1]
Saq[Saq<1e-6] = 1e-6
Sll = S_a1[0]*np.exp(S_a2[0]*zmid_ll)
Sll[0] = params[14]
Sll[Sll<1e-6] = 1e-6  
cll = -np.diff(zaq)[0::2] / (Kll*KvKh)
fhres   = 1.0/(params[12]*365.25) # fault entry resistance as d/Kfh with fault width nominally d = 1 m
fvres_m = 1.0/(params[13]*365.25) # flow resistance in fault (vertical) as d/Kfv with major fault length nominally d=1000m
fvres_s = 1.0/(params[13]*365.25) # flow resistance in fault (vertical) as d/Kfv with subseismic fault length nominally d=100m

#%% Create Baseline model
ml_b = ttim.ModelMaq(kaq=Kaq[0],
				   z=zaq[0:3],
				   c=cll[0],
				   Saq=Saq[0],
				   Sll=Sll[0], 
				   topboundary='leaky',
				   phreatictop=True,
				   tmin = .01,
				   tmax = outtime.max(),
				   M = 20 )
# Add basin boundary as noflow boundary
ttim.LeakyLineDoubletString(ml_b,
                            xy=boundary,
                            res='imp',
                            order=0,
                            layers=1,
                            label='BasinBoundary')
# Add mines
for i in range(nmines_b):
	mine_tsandQ = mine_q_b[mine_q_b[:,0]==np.unique(mine_q_b[:,0])[i],1::]
	mine_xy = mine_p_b[mine_p_b[:,0]==np.unique(mine_p_b)[i],1::]
	ttim.MscreenLineSinkDitchString( ml_b, 
									 xy=mine_xy, 
									 tsandQ=mine_tsandQ, 
									 res=0.1, wh='H',layers=1,
									 label='Mine_'+str(i+1) )
# solve model
ml_b.solve()

#%% write output
h_gwreceptors = np.zeros((len(gwreceptors),len(outtime)))
for i in range(len(gwreceptors)):
	h_gwreceptors[i,:] = ml_b.head(gwreceptors[i,0],
								   gwreceptors[i,1],
								   outtime,layers=1)
x_str = np.char.mod('%10.6e',np.concatenate((gwreceptors,h_gwreceptors),axis=1))
with open(fnames[0],'w') as f:
    f.write('ElementID, Easting, Northing,'+','.join(np.char.mod('t=%i',outtime))+'\n')
    for i in range(len(x_str)):
        f.write(gwID[i]+','+','.join(x_str[i])+'\n')

np.savetxt(fnames[0],
		   np.concatenate((gwreceptors,h_gwreceptors),axis=1),
		   delimiter = ',',
		   header = 'Easting,Northing,'+','.join(np.char.mod('t=%i',outtime)),
		   fmt = '%10.2f,%10.2f'+len(outtime)*',%10.5e' )

h_targets = np.ones((len(gwtargets),len(outtime)))
for i in range(len(h_targets)):
	h_targets[i,:] = ml_b.head(gwtargets[i,0],gwtargets[i,1],outtime,layers=1)
np.savetxt(fnames[1],
		   np.concatenate((gwtargets,h_targets),axis=1),
		   delimiter = ',',
		   header = 'Easting,Northing,'+','.join(np.char.mod('t=%i',outtime)),
		   fmt = '%10.2f,%10.2f'+len(outtime)*',%10.5e' )

#%% Create CRDP model
ml_c1 = ttim.ModelMaq(kaq=Kaq,
				   z=zaq,
				   c=cll,
				   Saq=Saq,
				   Sll=Sll,
				   topboundary='leaky',
				   phreatictop=True,
				   tmin = .01,
				   tmax = wells[:,4].max(),
				   M = 15 )
# Add basin boundary as noflow boundary
ttim.LeakyLineDoubletString(ml_c1,
                            xy=boundary,
                            res='imp',
                            order=0,
                            layers=1,
                            label='BasinBoundary')
# Add wells
for i in range(nwells):
	well_layers = np.argwhere(wells[i,5::]>0)+2
	ttim.HeadWell(ml_c1,
				  xw=wells[i,1],
				  yw=wells[i,2],
				  rw=.25,
				  tsandh = [wells[i,3],zmid_aq[well_layers[0]]+25],
				  res=0,
				  layers=well_layers.flatten(),
				  label='well_'+str(int(wells[i,0])) )
# Add major faults
for i in range(nfaults):
	xy_f = majorfaults[majorfaults[:,0]==np.unique(majorfaults[:,0])[i],1:3]
	ttim.ZeroMscreenLineSinkString(ml_c1,xy=xy_f,res=fhres,wh='H',
								   layers=range(naq),vres=fvres_m,wv=.1,
								   label='Fault_v_'+str(i))
# Add subseismic faults
for i in range(nsubfaults):
	xy_f = subfaults[subfaults[:,0]==i+1,1:4]
	cs_f = subfaults[subfaults[:,0]==i+1,3][0]-2
	ttim.ZeroMscreenLineSinkString(ml_c1,xy=xy_f,res=fhres,wh='H',
								   layers=range(int(cs_f),int(cs_f+4)),vres=fvres_s,wv=.1,
								   label='SubFault_v_'+str(i))
# Add mines
for i in range(nmines_c):
	mine_tsandQ = mine_q_c[mine_q_c[:,0]==np.unique(mine_q_c[:,0])[i],1::]
	mine_xy = mine_p_c[mine_p_c[:,0]==np.unique(mine_p_c)[i],1::]
	ttim.MscreenLineSinkDitchString( ml_c1, 
									 xy=mine_xy, 
									 tsandQ=mine_tsandQ, 
									 res=0.1, wh='H',layers=1,
									 label='Mine_'+str(i+1) )
# solve model
ml_c1.solve()
# compute water production curve
wprodlen = 20
Q = np.zeros( ( (wprodlen),2,nwells) )
for i in range(nwells):
	Q[0:wprodlen,0,i] = np.linspace(wells[i,3],wells[i,4],wprodlen) 
	Q[0:wprodlen-1,1,i] = ml_c1.strength('well_'+str(int(wells[i,0])),Q[1:wprodlen,0,i])[0,:]

#%% recreate model, wells now specified flux
## Create model
ml_c2 = ttim.ModelMaq(kaq=Kaq,
				   z=zaq,
				   c=cll,
				   Saq=Saq,
				   Sll=Sll,
				   topboundary='leaky',
				   phreatictop=False,
				   tmin = .1,
				   tmax = outtime.max(),
				   M = 15 )
# Add basin boundary as noflow boundary
ttim.LeakyLineDoubletString(ml_c2,
                            xy=boundary,
                            res='imp',
                            order=0,
                            layers=1,
                            label='BasinBoundary')
# Add wells
for i in range(nwells):
	timeq = Q[:,:,i]
	well_layers = np.argwhere(wells[i,5::]>0)+2
	ttim.Well(ml_c2,
			  xw=wells[i,1],
			  yw=wells[i,2],
			  rw=0.25,
			  tsandQ = timeq,
			  res=0.0,
			  layers=well_layers.flatten(),
			  label='well'+str(i))
# Add major faults
for i in range(nfaults):
	xy_f = majorfaults[majorfaults[:,0]==np.unique(majorfaults[:,0])[i],1:3]
	ttim.ZeroMscreenLineSinkString(ml_c2,xy=xy_f,res=fhres,wh='H',
								   layers=range(naq),vres=fvres_m,wv=0.1,
								   label='Fault_v_'+str(i))
# Add subseismic faults
for i in range(nsubfaults):
	xy_f = subfaults[subfaults[:,0]==i+1,1:4]
	cs_f = subfaults[subfaults[:,0]==i+1,3][0]-2
	ttim.ZeroMscreenLineSinkString(ml_c2,xy=xy_f,res=fhres,wh='H',
								   layers=range(int(cs_f),int(cs_f+4)),vres=fvres_s,wv=.1,
								   label='SubFault_v_'+str(i))
# Add mines
for i in range(nmines_c):
	mine_tsandQ = mine_q_c[mine_q_c[:,0]==np.unique(mine_q_c[:,0])[i],1::]
	mine_xy = mine_p_c[mine_p_c[:,0]==np.unique(mine_p_c)[i],1::]
	ttim.MscreenLineSinkDitchString( ml_c2, 
									 xy=mine_xy, 
									 tsandQ=mine_tsandQ, 
									 res=0.1, wh='H',layers=1,
									 label='Mine_'+str(i+1) )
# solve model
ml_c2.solve()

#%% Compute outputs
# groundwater receptor locations
h_gwreceptors = np.zeros((len(gwreceptors),len(outtime)))
for i in range(len(gwreceptors)):
	h_gwreceptors[i,:] = ml_c2.head(gwreceptors[i,0],
									gwreceptors[i,1],
									outtime,layers=1)
x_str = np.char.mod('%10.6e',np.concatenate((gwreceptors,h_gwreceptors),axis=1))
with open(fnames[2],'w') as f:
    f.write('ElementID, Easting, Northing,'+','.join(np.char.mod('t=%i',outtime))+'\n')
    for i in range(len(x_str)):
        f.write(gwID[i]+','+','.join(x_str[i])+'\n')

# interpolation points for modflow alluvial models
h_targets = np.ones((len(gwtargets),len(outtime)))
for i in range(len(h_targets)):
	h_targets[i,:] = ml_c2.head(gwtargets[i,0],gwtargets[i,1],outtime,layers=1)
np.savetxt(fnames[3],
		   np.concatenate((gwtargets,h_targets),axis=1),
		   delimiter = ',',
		   header = 'Easting,Northing,'+','.join(np.char.mod('t=%i',outtime)),
		   fmt = '%10.2f,%10.2f'+len(outtime)*',%10.5e' )
# water production curve CSG wells
CSG_prod = np.sum(Q,axis=2)
CSG_prod[:,0] = CSG_prod[:,0]/nwells
np.savetxt(fnames[4], CSG_prod, 
		   delimiter =',',
		   header='Time (Years),Q (m3/yr)',
		   fmt='%10.2f,%10.2f' )
