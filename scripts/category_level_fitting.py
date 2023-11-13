#%%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.linear_regression_model import MainTraining, KFoldCV

#%%
# prepare data
# read betas
betas_dir = '/home1/data/common-data/natural-scenes-dataset/rsa/roi_analyses/'
targetspace = 'fsaverage'
n_sub = 8
sub_list = [f'subj0{i+1}' for i in range(n_sub)]

sub = sub_list[0]
betas_mean_file = os.path.join(
            betas_dir, f'{sub}_betas_list_{targetspace}_averaged.npy'
    )

betas = np.load(betas_mean_file) #(327684, 10000)

# %%    
category_mat = pd.read_csv('../data/category_mat_original.csv', index_col=0).values.astype('float64')
# extract subject specific rows (consitions)

sum = np.sum(category_mat, axis=1).reshape(-1, 1)
category_mat /= sum # regularize each row
category_mat = torch.Tensor(category_mat)