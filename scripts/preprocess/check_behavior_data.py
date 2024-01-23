#%%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
# load data
data_dir = '../../data'
behavior_data = pd.read_csv(os.path.join(data_dir, 'behavior/sis_all_dissim.csv'), index_col=0)

# get matrix
behavior_mat = behavior_data.pivot(index='image1', columns='image2', values='agg_score')
# %%
# calculate the number of Nan
behavior_mat.isna().sum().sum()

# caculate the number of not Nan
behavior_mat.notna().sum().sum()
# %%
# calculate the max number of not Nan in a row
behavior_mat.notna().sum(axis=1).max()
# %%

# load category mat
category_mat = pd.read_csv(os.path.join(data_dir, 'behavior/category_mat_behavior.csv'), index_col=0)

# convert the index and columns to interger indices
category_mat.index = range(len(category_mat.index))
category_mat.columns = range(len(category_mat.columns))
# %%

# convert the image numbers of behavior data into the indices of category mat
correspondence_dict = category_mat.idxmax(axis=1).to_dict()

def convert_to_index(value):
    return correspondence_dict.get(value, None)
# %%
behavior_data['image1'] = behavior_data['image1'].apply(convert_to_index)
behavior_data['image2'] = behavior_data['image2'].apply(convert_to_index)
# %%
behavior_data.to_csv(os.path.join(data_dir, 'behavior/sis_all_dissim_category.csv'))
# %%
# convert behavior data to matrix
# if index contains duplicate entries, use pivot_table

behavior_mat = behavior_data.pivot_table(index='image1', columns='image2', values='agg_score')
# calculate the number of Nan
behavior_mat.isna().sum().sum()

# %%
