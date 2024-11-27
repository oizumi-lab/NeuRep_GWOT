#%%
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import time
import numpy as np
import nibabel as nib
from scipy.spatial.distance import pdist, cdist
from nsd_access.nsd_access import NSDAccess
from nsddatapaper_rsa.utils.nsd_get_data import get_conditions, get_betas
from nsddatapaper_rsa.utils.utils import average_over_conditions
from nsdcode.nsd_mapdata import NSDmapdata

from src.utils import sample_participants, split_lists, show_matrix

#%%
"""
    module to gather the region of interest rdms
"""

n_jobs = 38
n_sessions = 40
n_subjects = 8
# subjects
subs = ['subj0{}'.format(x+1) for x in range(n_subjects)]

#%%
#%%
n_subj = 8
n_groups = 2
subj_list = subs
n_sample = 10
seed_list = range(n_sample)

# subjects groups for each seed
groups_list = []
for seed in seed_list:
    subj_list = sample_participants(n_subj, n_subj, seed)
    groups = split_lists(subj_list, n_groups)
    groups_list.append(groups)

roi_categories = ["highlevelvisual"]#, "streams""floc-places", "MTL", 
# ROIS = {1: 'pVTC', 2: 'aVTC', 3: 'v1', 4: 'v2', 5: 'v3'}
#ROIS = {7: 'hV4'}
#ROIS = {1: "LGN", 2: "ventralPul", 3: "dorsolateralPul", 4: "dorsomedialPul", 5: "SC"}
#ROIS = {1: "thalamus"}
# ROIS = {1: "MTL"}
#ROIS = {1: "early", 2: "midventral", 3: "midlateral", 4: "midparietal", 5: "ventral", 6: "lateral", 7: "parietal"}
# ROIS = {1: "OPA", 2: "PPA", 3: "RSC"} 

# we use the fsaverage space.
# targetspace = 'func1pt8mm' # 'func1pt8mm''fsaverage'
# targetspace = 'fsaverage'

#%%
for roi_category in roi_categories:
    if roi_category == "highlevelvisual":
        ROIS = {1: 'pVTC', 2: 'aVTC', 3: 'v1', 4: 'v2', 5: 'v3'}
        targetspace = 'fsaverage'
    elif roi_category == "floc-places":
        ROIS = {1: 'OPA', 2: 'PPA', 3: 'RSC'}
        targetspace = 'fsaverage'
    elif roi_category == "streams":
        ROIS = {1: "early", 2: "midventral", 3: "midlateral", 4: "midparietal", 5: "ventral", 6: "lateral", 7: "parietal"}
        targetspace = 'fsaverage'
    elif roi_category == "MTL":
        ROIS = {1: "MTL"}
        targetspace = 'func1pt8mm'

    for seed, groups in enumerate(groups_list):
        for group_idx, group in enumerate(groups):
            print(f'Processing seed {seed}, group {group_idx}')
            
            group_betas_shared515 = []
            group_mask_data = []
            for i in group:
                sub = subs[i]
                # set up directories
                base_dir = "/mnt/NAS/common_data/natural-scenes-dataset/"
                nsd_dir = base_dir
                proj_dir = base_dir
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

                if targetspace == 'fsaverage':
                    if roi_category == "highlevelvisual":
                        lh_file = os.path.join("../../nsddatapaper/mainfigures/SCIENCE.RSA", 'lh.highlevelvisual.mgz')
                        rh_file = os.path.join("../../nsddatapaper/mainfigures/SCIENCE.RSA", 'rh.highlevelvisual.mgz')
                    
                    if roi_category == "floc-places":
                        lh_file = os.path.join(nsd_dir, 'nsddata', 'freesurfer', f'{sub}', 'label', 'lh.floc-places.mgz')
                        rh_file = os.path.join(nsd_dir, 'nsddata', 'freesurfer', f'{sub}', 'label', 'rh.floc-places.mgz')
                    if roi_category == "streams":
                        lh_file = os.path.join(nsd_dir, 'nsddata', 'freesurfer', f'{sub}', 'label', 'lh.streams.mgz')
                        rh_file = os.path.join(nsd_dir, 'nsddata', 'freesurfer', f'{sub}', 'label', 'rh.streams.mgz')
                    
                    
                    if roi_categories == "highlevelvisual":
                        # load the lh mask
                        maskdata_lh = nib.load(lh_file).get_fdata().squeeze()
                        maskdata_rh = nib.load(rh_file).get_fdata().squeeze()
                        
                        maskdata = np.hstack((maskdata_lh, maskdata_rh))
                    
                    else:
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
                
                elif targetspace == 'func1pt8mm':
                    if roi_category == "MTL":
                        file = os.path.join(nsd_dir, 'nsddata', 'ppdata', f'{sub}', f'{targetspace}', 'roi', 'MTL.nii.gz')
                        maskdata = nib.load(file).get_fdata().squeeze()
                    elif roi_category == "thalamus":
                        file = os.path.join(nsd_dir, 'nsddata', 'ppdata', f'{sub}', f'{targetspace}', 'roi', 'thalamus.nii.gz')
                        maskdata = nib.load(file).get_fdata().squeeze()
                
                group_mask_data.append(maskdata)
                
                # extract conditions
                conditions = get_conditions(nsd_dir, sub, n_sessions)
                conditions = np.asarray(conditions).ravel()
                conditions_bool = [
                    True if np.sum(conditions == x) == 3 else False for x in conditions]
                sample_shared515 = np.array(np.load("../../data/shared515ids.npy"))

                betas_mean_shared515_file = os.path.join(
                        outpath, f'{sub}_betas_list_{targetspace}_averaged_shared515.npy'
                )

                if not os.path.exists(betas_mean_shared515_file):
                    betas_mean_shared515 = get_betas(
                        nsd_dir,
                        sub,
                        n_sessions,
                        targetspace=targetspace,
                    )
                    print(f'concatenating betas for {sub}')
                    
                    n_dim = len(betas_mean_shared515[0].shape)
                    if n_dim == 2:
                        betas_mean_shared515 = np.concatenate(betas_mean_shared515, axis=1).astype(np.float32)
                    elif n_dim == 4:
                        betas_mean_shared515 = np.concatenate(betas_mean_shared515, axis=3).astype(np.float32)

                    print(f'averaging betas for {sub}')
                    betas_mean_shared515 = average_over_conditions(
                        betas_mean_shared515,
                        conditions,
                        sample_shared515,
                    ).astype(np.float32)

                    print(f'saving condition averaged betas for {sub}')
                    np.save(betas_mean_shared515_file, betas_mean_shared515)
                else:
                    print(f'loading betas for {sub}')
                    betas_mean_shared515 = np.load(betas_mean_shared515_file, allow_pickle=True)
                    n_dim = len(betas_mean_shared515[0].shape)

                group_betas_shared515.append(betas_mean_shared515)

            # axis = 0 if targetspace == 'fsaverage' else 3
            # group_mask_data = np.concatenate(group_mask_data, axis=axis)

            for roi, mask_name in ROIS.items():
                print(f'working on ROI: {mask_name}')
                masked_betas_shared515 = []
                for betas, maskdata in zip(group_betas_shared515, group_mask_data):
                    if mask_name == "thalamus" or mask_name == "MTL":
                        vs_mask = maskdata >= 1
                    else:
                        vs_mask = maskdata == roi
                    masked_betas_shared515.append(betas[vs_mask, :])
                
                masked_betas_shared515 = np.concatenate(masked_betas_shared515, axis=0)
                print(masked_betas_shared515.shape)

                rdm_file_shared515 = os.path.join(
                    outpath, f'seed{seed}_group{group_idx}_{mask_name}_fullrdm_shared515_correlation.npy'
                )

                check_nans = False
                if check_nans:
                    good_vox = [
                        True if np.sum(
                            np.isnan(x)
                            ) == 0 else False for x in masked_betas_shared515]

                    if np.sum(good_vox) != len(good_vox):
                        print(f'found some NaN for ROI: {mask_name} - {sub}')

                    masked_betas_shared515 = masked_betas_shared515[good_vox, :]

                else:
                    masked_betas_shared515 = np.nan_to_num(masked_betas_shared515)

                X = masked_betas_shared515.T

                print(f'computing RDM for roi: {mask_name}')
                start_time = time.time()
                
                rdm = cdist(X, X, metric='correlation')

                if check_nans:
                    if np.any(np.isnan(rdm)):
                        raise ValueError
                else:
                    if np.any(np.isnan(rdm)):
                        print(f'found some NaN for ROI: {mask_name} - {sub}')
                        rdm = np.nan_to_num(rdm)

                elapsed_time = time.time() - start_time
                print(
                    'elapsedtime: ',
                    f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
                )
                print(f'saving full rdm for {mask_name} : seed {seed}, group {group_idx}')
                np.save(
                    rdm_file_shared515,
                    rdm
                )