#%%
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

#%%
n_sessions = 40
n_subjects = 8
# subjects
subs = ['subj0{}'.format(x+1) for x in range(n_subjects)]

#%%
for sub in subs:
    # set up directories
    #nsd_dir = "/home1/common-data/natural-scenes-dataset/"
    base_dir = "/home1/data/common-data/natural-scenes-dataset/"
    nsd_dir = base_dir
    proj_dir = base_dir
    #nsd_dir = os.path.join(base_dir, 'charesti-start', 'data', 'NSD')
    sem_dir = os.path.join(proj_dir, 'derivatives', 'ecoset')
    betas_dir = os.path.join(proj_dir, 'rsa')
    models_dir = os.path.join(proj_dir, 'rsa', 'serialised_models')
    
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
    
    np.save(f'../data/unique_conditions_{sub}', sample)
# %%
