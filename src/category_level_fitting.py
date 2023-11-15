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
def load_betas(sub, betas_dir, targetspace):
    # load betas and return it as a torch.tensor
    
    betas_mean_file = os.path.join(
                betas_dir, f'{sub}_betas_list_{targetspace}_averaged.npy'
        )

    betas = np.load(betas_mean_file)

    betas = betas.T
    betas = torch.Tensor(betas) # shape of ([10000, 327684])
    
    return betas

def load_category_mat(sub):
    # load the category matrix and return it as a torch.tensor
    
    category_mat = pd.read_csv('../data/category_mat_original.csv', index_col=0)

    # extract subject specific rows (consitions)
    conditions = np.load(f'../data/unique_conditions_{sub}.npy')
    conditions -= 1

    category_mat = category_mat.loc[conditions].values.astype('float64')
    sum = np.sum(category_mat, axis=1).reshape(-1, 1)
    category_mat /= sum # regularize each row
    category_mat = torch.Tensor(category_mat) # shape of ([10000, 80])
    
    return category_mat
# %%
if __name__ == '__main__':
    ### prepare data
    data = 'betas'
    
    betas_dir = '/home1/data/common-data/natural-scenes-dataset/rsa/roi_analyses/'
    targetspace = 'fsaverage'
    n_sub = 8
    sub_list = [f'subj0{i+1}' for i in range(n_sub)]

    sub = sub_list[0]
    study_name = f"fitting_to_{data}_{sub}"

    betas = load_betas(sub=sub, betas_dir=betas_dir, targetspace=targetspace)
    category_mat = load_category_mat(sub=sub)

    #%%
    ### fitting
    dataset = TensorDataset(category_mat, betas)

    ### set parameters
    # device 
    #device = 'cpu'
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # Train the model
    n_category = 80
    n_emb = betas.shape[1] # number of voxels (e.g., 327684)

    # Define the optimizer (e.g., Adam)
    lr = 0.01
    num_epochs = 100
    early_stopping = True
    reg='l2'
    batch_size = 3

    loss_fn = nn.MSELoss()

    ### cv params
    n_splits = 5
    lamb_range = [1e-3, 1e-1]
    cv_n_trial = 1
    # %%
    #cv = KFoldCV(
    #    dataset=dataset,
    #    n_splits=n_splits,
    #    search_space=lamb_range,
    #    study_name=study_name,
    #    results_dir='../results/cv_fitting/betas/',
    #    batch_size=batch_size,
    #    device=device,
    #    loss_fn=loss_fn,
    #    n_category=n_category,
    #    n_emb=n_emb,
    #    n_epoch=num_epochs,
    #    lr=lr,
    #    reg=reg,
    #    early_stopping=early_stopping,
    #)
    #
    #cv.optimize(n_trials=cv_n_trial)
    
    ### main
    lamb = 1e-3
    
    main_training = MainTraining(
        dataset = dataset, 
        test_size = 1/n_splits, 
        batch_size = batch_size, 
        device = device
        )
    
    embeddings, loss = main_training.main_compute(
        loss_fn = loss_fn, 
        n_emb = n_emb, 
        n_category = n_category, 
        n_epoch = num_epochs, 
        lr = lr, 
        early_stopping=early_stopping,
        reg=reg, 
        lamb=lamb
        )
    
    np.save(f'../data/embeddings/{data}/category_level_embeddings_{sub}.', embeddings)
    
    #%%