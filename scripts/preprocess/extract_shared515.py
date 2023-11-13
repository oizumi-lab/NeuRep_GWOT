#%%
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import time
import numpy as np
import nibabel as nib
import shutil
from scipy.spatial.distance import pdist, cdist
from nsd_access.nsd_access import NSDAccess
from nsddatapaper_rsa.utils.nsd_get_data import get_conditions, get_betas
from nsddatapaper_rsa.utils.utils import average_over_conditions

from functools import reduce

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
samples = []
for sub in subs[:1]:

    # set up directories
    #nsd_dir = "/home1/common-data/natural-scenes-dataset/"
    base_dir = "/home1/data/common-data/natural-scenes-dataset/"
    nsd_dir = base_dir
    proj_dir = base_dir
    #nsd_dir = os.path.join(base_dir, 'charesti-start', 'data', 'NSD')
    sem_dir = os.path.join(proj_dir, 'derivatives', 'ecoset')
    betas_dir = os.path.join(proj_dir, 'rsa')
    models_dir = os.path.join(proj_dir, 'rsa', 'serialised_models')

    # initiate nsd access
    nsda = NSDAccess(nsd_dir)

    # path where we save the rdms
    outpath = os.path.join(betas_dir, 'roi_analyses')
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # we use the fsaverage space.
    targetspace = 'fsaverage'

    lh_file = os.path.join("../nsddatapaper/mainfigures/SCIENCE.RSA", 'lh.highlevelvisual.mgz')
    rh_file = os.path.join("../nsddatapaper/mainfigures/SCIENCE.RSA", 'rh.highlevelvisual.mgz')

    # load the lh mask
    maskdata_lh = nib.load(lh_file).get_fdata().squeeze()
    maskdata_rh = nib.load(rh_file).get_fdata().squeeze()

    maskdata = np.hstack((maskdata_lh, maskdata_rh))

    ROIS = {1: 'pVTC', 2: 'aVTC', 3: 'v1', 4: 'v2', 5: 'v3'}

    roi_names = ['pVTC', 'aVTC', 'v1', 'v2', 'v3']

    # sessions
    n_sessions = 40

    # subjects
    subs = ['subj0{}'.format(x+1) for x in range(n_subjects)]

    # extract conditions
    conditions = get_conditions(nsd_dir, sub, n_sessions)

    # we also need to reshape conditions to be ntrials x 1
    conditions = np.asarray(conditions).ravel()

    # then we find the valid trials for which we do have 3 repetitions.
    conditions_bool = [
        True if np.sum(conditions == x) == 3 else False for x in conditions]

    conditions_sampled = conditions[conditions_bool]

    # find the subject's unique condition list (sample pool)
    sample = np.unique(conditions[conditions_bool])
    
    samples.append(sample)
    
shared_515 = reduce(np.intersect1d, samples)

# %%
print(len(shared_515))
# %%
np.save("../data/shared515ids", shared_515) # indices start from 1
# %%
# extract images of shared 515
data_path = '/home1/data/common-data/natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/'
shared_515 = np.load("../data/shared515ids.npy", allow_pickle=True)

shared_515 -= 1

for i in shared_515:
    source_file = os.path.join(data_path, f'image{i}/image{i}.png')
    destination_file = os.path.join(data_path, 'shared_515', f'image{i}/image{i}.png')
    
    os.makedirs(os.path.join(data_path, 'shared_515', f'image{i}/'), exist_ok=True)
    shutil.copy(source_file, destination_file)
# %%
