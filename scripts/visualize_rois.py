#%%
from surfer import Brain
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import time
import numpy as np
import nibabel as nib
from scipy.spatial.distance import pdist, cdist
from nsd_access.nsd_access import NSDAccess
from nsddatapaper_rsa.utils.nsd_get_data import get_conditions, get_betas
from nsddatapaper_rsa.utils.utils import average_over_conditions
from nsdcode.nsd_mapdata import NSDmapdata

targetspace = 'fsaverage'
ROIS = {1: "early", 2: "midventral", 3: "midlateral", 4: "midparietal", 5: "ventral", 6: "lateral", 7: "parietal"}

# subjects
n_subjects = 8
subs = ['subj0{}'.format(x+1) for x in range(n_subjects)]

base_dir = "/mnt/NAS/common_data/natural-scenes-dataset/"
nsd_dir = base_dir

subject_dir = "/mnt/NAS/common_data/natural-scenes-dataset/nsddata/freesurfer"
#%%
nsda = NSDAccess(nsd_dir)
mapdata = NSDmapdata(nsd_dir)

#%%
for i, sub in enumerate(subs[:1]):
    #lh_file = os.path.join(nsd_dir, 'nsddata', 'freesurfer', f'{sub}', 'label', 'lh.streams.mgz') #prf-visualrois
    #rh_file = os.path.join(nsd_dir, 'nsddata', 'freesurfer', f'{sub}', 'label', 'rh.streams.mgz')
    
    lh_file = os.path.join(nsd_dir, 'nsddata', 'ppdata', f'{sub}', 'func1pt8mm', 'roi', 'lh.streams.nii.gz')
    rh_file = os.path.join(nsd_dir, 'nsddata', 'ppdata', f'{sub}', 'func1pt8mm', 'roi', 'rh.streams.nii.gz')
    
    
    # load the lh mask
    maskdata_lh = nib.load(lh_file).get_fdata().squeeze()
    maskdata_rh = nib.load(rh_file).get_fdata().squeeze()
    
    #maskdata_lh = mapdata.fit(
    #    subjix=i+1,
    #    sourcespace='lh.white',
    #    targetspace='fsaverage',
    #    sourcedata=maskdata_lh,
    #    outputclass='single',
    #    outputfile=None,
    #    interptype='nearest'
    #)
    #
    #maskdata_rh = mapdata.fit(
    #    subjix=i+1,
    #    sourcespace='rh.white',
    #    targetspace='fsaverage',
    #    sourcedata=maskdata_rh,
    #    outputclass='single',
    #    outputfile=None,
    #    interptype='nearest'
    #)
    
    brain_lh = Brain('fsaverage', 'lh', 'inflated', background='white', subjects_dir=subject_dir)
    brain_rh = Brain('fsaverage', 'rh', 'inflated', background='white', subjects_dir=subject_dir)
    
    # add the mask to the brain
    brain_lh.add_data(maskdata_lh, 0, 7, colormap='Set1', alpha=.8, colorbar=False)
    brain_rh.add_data(maskdata_rh, 0, 7, colormap='Set1', alpha=.8, colorbar=False)
# %%
