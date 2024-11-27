#%%
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import time
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.spatial.distance import pdist, cdist
from nsd_access.nsd_access import NSDAccess
from nsddatapaper_rsa.utils.nsd_get_data import get_conditions, get_betas
from nsddatapaper_rsa.utils.utils import average_over_conditions
from nsdcode.nsd_mapdata import NSDmapdata
import seaborn as sns

#%%
n_jobs = 38
n_sessions = 40
n_subjects = 8
# subjects
subs = ['subj0{}'.format(x+1) for x in range(n_subjects)]



ROIS = {1: 'pVTC', 2: 'aVTC', 3: 'v1', 4: 'v2', 5: 'v3'}
#ROIS = {7: 'hV4'}
#ROIS = {1: "LGN", 2: "ventralPul", 3: "dorsolateralPul", 4: "dorsomedialPul", 5: "SC"}
#ROIS = {1: "thalamus"}
#ROIS = {1: "MTL"}
#ROIS = {1: "early", 2: "midventral", 3: "midlateral", 4: "midparietal", 5: "ventral", 6: "lateral", 7: "parietal"}

# we use the fsaverage space.
targetspace = 'func1pt8mm' # 'func1pt8mm''fsaverage'

files = ["MTL", "default", "floc-places"]#"streams", "thalamus", 
targetspaces = ["fsaverage", "func1pt8mm", "func1pt8mm"]

voxel_numbers = pd.DataFrame(columns=['sub', 'roi', 'voxel_number'])
#%%
for i, sub in enumerate(subs):

    # set up directories
    #nsd_dir = "/home1/common-data/natural-scenes-dataset/"
    #base_dir = "/home1/data/common-data/natural-scenes-dataset/"
    base_dir = "/mnt/NAS/common_data/natural-scenes-dataset/"
    nsd_dir = base_dir
    proj_dir = base_dir
    #nsd_dir = os.path.join(base_dir, 'charesti-start', 'data', 'NSD')
    sem_dir = os.path.join(proj_dir, 'derivatives', 'ecoset')
    betas_dir = os.path.join(proj_dir, 'rsa')
    models_dir = os.path.join(proj_dir, 'rsa', 'serialised_models')

    # initiate nsd access
    nsda = NSDAccess(nsd_dir)
    mapdata = NSDmapdata(nsd_dir)

    # path where we save the rdms
    outpath = os.path.join(betas_dir, 'roi_analyses')
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    #lh_file = os.path.join("../../nsddatapaper/mainfigures/SCIENCE.RSA", 'lh.highlevelvisual.mgz')
    #rh_file = os.path.join("../../nsddatapaper/mainfigures/SCIENCE.RSA", 'rh.highlevelvisual.mgz')

    for targetspace, file_name in zip(targetspaces, files):
        if targetspace == 'fsaverage':
            lh_file = os.path.join(nsd_dir, 'nsddata', 'freesurfer', f'{sub}', 'label', f'lh.{file_name}.mgz') #prf-visualrois
            rh_file = os.path.join(nsd_dir, 'nsddata', 'freesurfer', f'{sub}', 'label', f'rh.{file_name}.mgz')

            # load the lh mask
            maskdata_lh = nib.load(lh_file).get_fdata().squeeze()
            maskdata_rh = nib.load(rh_file).get_fdata().squeeze()

            maskdata_lh = mapdata.fit(
                subjix=i+1,
                sourcespace='lh.white',
                targetspace='fsaverage',
                sourcedata=maskdata_lh,
                outputclass='single',
                outputfile=None,
                interptype='nearest'
            )

            maskdata_rh = mapdata.fit(
                subjix=i+1,
                sourcespace='rh.white',
                targetspace='fsaverage',
                sourcedata=maskdata_rh,
                outputclass='single',
                outputfile=None,
                interptype='nearest'
            )

            maskdata = np.hstack((maskdata_lh, maskdata_rh))

        else:
            #lh_file = os.path.join(nsd_dir, 'nsddata', 'ppdata', f'{sub}', f'{targetspace}', 'roi', 'lh.thalamus.nii.gz')
            #rh_file = os.path.join(nsd_dir, 'nsddata', 'ppdata', f'{sub}', f'{targetspace}', 'roi', 'rh.thalamus.nii.gz')
#   
            ## load the lh mask
            #maskdata_lh = nib.load(lh_file).get_fdata().squeeze()
            #maskdata_rh = nib.load(rh_file).get_fdata().squeeze()
            file = os.path.join(nsd_dir, 'nsddata', 'ppdata', f'{sub}', f'{targetspace}', 'roi', f'{file_name}.nii.gz')
            maskdata = nib.load(file).get_fdata().squeeze()
            
        if file_name == "streams":
            ROIS = {1: "early", 2: "midventral", 3: "midlateral", 4: "midparietal", 5: "ventral", 6: "lateral", 7: "parietal"}
        elif file_name == "thalamus":
            ROIS = {1: "thalamus"}
        elif file_name == "MTL":
            ROIS = {1: "MTL"}
        elif file_name == "default":
            ROIS = {1: 'pVTC', 2: 'aVTC', 3: 'v1', 4: 'v2', 5: 'v3'}
        elif file_name == "floc-places":
            ROIS = {1: "OPA", 2: "PPA", 3: "RSC"} 
        
        # check numbers of voxels
        for roi, mask_name in ROIS.items():
            if mask_name == "thalamus" or mask_name == "MTL":
                vs_mask = maskdata >= 1
            else:
                # logical array of mask vertices
                vs_mask = maskdata == roi
            
            # number of vertices
            n_voxels = np.sum(vs_mask)
            voxel_numbers = voxel_numbers.append({'sub': sub, 'roi': mask_name, 'voxel_number': n_voxels}, ignore_index=True)

#%%
# save the voxel numbers
voxel_numbers.to_csv(os.path.join('../../data/voxel_numbers.csv'), index=False)
# %%
# plot the voxel numbers for each roi
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.barplot(x="roi", y="voxel_number", hue="sub", data=voxel_numbers)
plt.title("Voxel Numbers for Each ROI")
plt.xlabel("ROI")
plt.ylabel("Voxel Number")
plt.xticks(rotation=45)
plt.legend(title="Subject")
plt.show()
plt.savefig('../../results/figs/voxel_numbers.png')
# %%
