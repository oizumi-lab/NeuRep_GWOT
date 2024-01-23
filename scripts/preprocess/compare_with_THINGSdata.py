#%%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

#%%
# load category data
df_THINGS = pd.read_csv('../../data/THINGS/category_mat_manual_preprocessed.csv', index_col=0)
cat_THINGS = df_THINGS.columns.values

with open('../../data/categories.json', 'r') as f:
    cat_nsd = json.load(f)
print(cat_nsd)

cat_nsd515 = pd.read_csv('../../data/category_mat_shared515.csv', index_col=0)

# %%
cat_nsd_super = []
for cat in cat_nsd:
    cat_nsd_super.append(cat['supercategory'])
# %%
cat_nsd_super = np.unique(cat_nsd_super)
# %%
# check overlap
supcat_overlap = np.intersect1d(cat_THINGS, cat_nsd_super).tolist()

supcat_overlap_nsd = supcat_overlap + ['accessory', 'sports', 'kitchen', 'appliance', 'electronic']
supcat_overlap_THINGS = supcat_overlap + ['clothing accessory',  'sports equipment', 'kitchen appliance', 'kitchen tool', 'electronic device']

cat_overlap_nsd = []
for sup_cat in supcat_overlap_nsd:
    # extract categories corresponding to supercategory
    for cat in cat_nsd:
        if cat['supercategory'] == sup_cat:
            cat_overlap_nsd.append((sup_cat, cat['name']))
# %%
# check numbers
num_overlap = 0
for supcat, cat in cat_overlap_nsd:
    print(supcat, cat, cat_nsd515[cat].sum())
    num_overlap += cat_nsd515[cat].sum()
print(num_overlap)
# %%
# again, check overlap numbers on the THINGS side
num_overlap = 0
for cat in supcat_overlap_THINGS:
    print(cat, df_THINGS[cat].sum())
    num_overlap += df_THINGS[cat].sum()
print(num_overlap)

# %%
