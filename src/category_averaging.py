#%%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import pandas as pd
import numpy as np
from tqdm import tqdm
#%%
def load_category_mat(sub):
    # load the category matrix and return it as a torch.tensor
    
    category_mat = pd.read_csv('../data/category_mat_original.csv', index_col=0)

    # extract subject specific rows (consitions)
    conditions = np.load(f'../data/unique_conditions_{sub}.npy')
    conditions -= 1

    category_mat = category_mat.loc[conditions].values.astype('float64')
    sum = np.sum(category_mat, axis=1).reshape(-1, 1)
    category_mat /= sum # regularize each row 
    
    return category_mat # shape of ([10000, 80])
# %%
if __name__ == '__main__':
    ### prepare data
    data = 'betas'
    
    betas_dir = '/home1/data/common-data/natural-scenes-dataset/rsa/roi_analyses/'
    targetspace = 'fsaverage'
    n_sub = 8
    sub_list = [f'subj0{i+1}' for i in range(n_sub)]

    #sub = sub_list[0]
    for sub in tqdm(sub_list):
        betas_mean_file = os.path.join(
                    betas_dir, f'{sub}_betas_list_{targetspace}_averaged.npy'
            )

        avg_betas_file = os.path.join(
                    betas_dir, f'{sub}_betas_list_{targetspace}_category_averaged.npy'
            )

        betas = np.load(betas_mean_file) # shape of ([327684, 10000]) 

        category_mat = load_category_mat(sub=sub) # shape of ([10000, 80])

        #%%
        ### average for each category
        avg_betas = np.matmul(betas, category_mat) # shape of ([10000, 80])
        np.save(avg_betas_file, avg_betas)
    #%%