#%%
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import shutil
from tqdm import tqdm
from collections import defaultdict
import nltk
from nltk.corpus import wordnet as wn
from functools import reduce
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, cdist
import nibabel as nib
from nsd_access.nsd_access import NSDAccess
from nsddatapaper_rsa.utils.nsd_get_data import get_conditions, get_betas
from nsddatapaper_rsa.utils.utils import average_over_conditions


#%%
# Configuration
n_sessions = 40
n_subjects = 8
subs = ['subj0{}'.format(x+1) for x in range(n_subjects)]
nsd_dir = "/home1/data/common-data/natural-scenes-dataset/"
betas_dir = "/home1/data/common-data/natural-scenes-dataset/rsa/"
outpath = "/home1/data/common-data/natural-scenes-dataset/rsa/"

#%%
# Step 1: Prepare conditions
save_stim = os.path.join(outpath, 'all_subs_stims_full.npy')
if not os.path.exists(save_stim):
    conditions_all = []
    for sub in subs:
        conditions = get_conditions(nsd_dir, sub, n_sessions)
        conditions = np.asarray(conditions).ravel()
        conditions_all.append(conditions)

    conditions_all = np.concatenate(conditions_all, axis=0)
    conditions = np.unique(conditions_all)
    np.save(save_stim, conditions)
else:
    conditions = np.load(save_stim, allow_pickle=True)

#%%
# Step 2: Extract category labels
nsda = NSDAccess(nsd_dir)
categories = nsda.read_image_coco_category(conditions - 1)
categories = np.array(categories, dtype=object)
np.save(os.path.join(betas_dir, 'all_stims_category_labels.npy'), categories)

#%%
# Step 3: Process unique categories and vectors
category_label = np.load(os.path.join(betas_dir, 'all_stims_category_labels.npy'), allow_pickle=True)

# Get unique categories
unique_categories = set()
for sublist in category_label:
    unique_categories.update(sublist)
unique_categories = list(unique_categories)
np.save(os.path.join(betas_dir, 'all_stims_unique_category_labels.npy'), unique_categories)

# Get category vectors
vectors = []
for sublist in category_label:
    vector = np.zeros(len(unique_categories), dtype=int)
    for item in sublist:
        index = unique_categories.index(item)
        vector[index] = 1
    vectors.append(vector)
vectors = np.array(vectors)
np.save(os.path.join(betas_dir, 'all_stims_category_vectors.npy'), vectors)

#%%
# Step 4: Convert category vectors to max-zero columns
def convert_to_max_zero_columns(matrix):
    n, m = matrix.shape
    for i in tqdm(range(n)):
        if np.sum(matrix[i]) > 1:
            max_ones_col = np.argmax(np.sum(matrix, axis=0) * matrix[i])
            matrix[i] = np.where(np.arange(m) == max_ones_col, matrix[i], 0)
    for col in tqdm(range(m)):
        if np.sum(matrix[:, col]) == 0:
            matrix[:, col] = 0
    return matrix

category_vectors = np.load(os.path.join(betas_dir, 'all_stims_category_vectors.npy'))
category_mat = convert_to_max_zero_columns(category_vectors)

# Save category matrix
conditions -= 1
category_mat_df = pd.DataFrame(data=category_mat, columns=unique_categories, index=conditions)
category_mat_df.to_csv("../../data/category_mat.csv")

#%%
# Step 5: Sort and save category matrix
filename = '../../data/categories.json'
with open(filename, 'r') as file:
    data = json.load(file)
sorted_list = [cat['name'] for cat in data]
category_mat_df_sorted = category_mat_df.reindex(columns=sorted_list)
category_mat_df_sorted.to_csv("../../data/category_mat.csv")

# Save original matrix
original_mat = pd.DataFrame(data=category_vectors, columns=unique_categories, index=conditions)
original_mat = original_mat.reindex(columns=sorted_list)
original_mat.to_csv("../../data/category_mat_original.csv")

#%%
# Step 6: Extract shared 515
samples = []
for sub in subs[:1]:
    conditions = get_conditions(nsd_dir, sub, n_sessions)
    conditions = np.asarray(conditions).ravel()
    conditions_bool = [True if np.sum(conditions == x) == 3 else False for x in conditions]
    sample = np.unique(conditions[conditions_bool])
    samples.append(sample)
shared_515 = reduce(np.intersect1d, samples)
np.save("../../data/shared515ids", shared_515)

# Save shared 515 category matrix
shared_515 = np.load("../../data/shared515ids.npy", allow_pickle=True)
category_mat_shared515 = category_mat_df_sorted.loc[shared_515 - 1]
category_mat_shared515.to_csv('../../data/category_mat_shared515.csv')

#%%
# Step 7: Extract images of shared 515
data_path = '/home1/data/common-data/natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/'
shared_515 -= 1
for i in shared_515:
    source_file = os.path.join(data_path, f'image{i}/image{i}.png')
    destination_file = os.path.join(data_path, 'shared_515', f'image{i}/image{i}.png')
    os.makedirs(os.path.join(data_path, 'shared_515', f'image{i}/'), exist_ok=True)
    shutil.copy(source_file, destination_file)
# %%