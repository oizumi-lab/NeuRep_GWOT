#%%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import pandas as pd
import numpy as np
from tqdm import tqdm
import nibabel as nib
from scipy.spatial import distance

#%%
def load_category_mat(sub):
    # load the category matrix and return it as a torch.tensor
    
    category_mat = pd.read_csv('../data/category_mat_original.csv', index_col=0)

    # extract subject specific rows (consitions)
    conditions = np.load(f'../data/unique_conditions_{sub}.npy')
    conditions -= 1

    e = 1e-10
    category_mat = category_mat.loc[conditions].values.astype('float64')
    sum = np.sum(category_mat, axis=1).reshape(-1, 1)
    category_mat /= (sum + e) # normalize each row 
    
    sum = np.sum(category_mat, axis=0).reshape(1, -1)
    category_mat /= (sum + e) # normalize each column
    
    return category_mat # shape of ([10000, 80])
# %%
if __name__ == '__main__':
    ### prepare data
    data = 'betas'
    
    betas_dir = '/home1/data/common-data/natural-scenes-dataset/rsa/roi_analyses/'
    targetspace = 'fsaverage'
    n_sub = 8
    sub_list = [f'subj0{i+1}' for i in range(n_sub)]

    #%%
    #sub = sub_list[0]
    for sub in tqdm(sub_list):
        betas_mean_file = os.path.join(
                    betas_dir, f'{sub}_betas_list_{targetspace}_averaged.npy'
            )

        betas = np.load(betas_mean_file) # shape of ([327684, 10000]) 

        category_mat = load_category_mat(sub=sub) # shape of ([10000, 80])

        ### get RDMS                
        ROIS = {1: 'pVTC', 2: 'aVTC', 3: 'v1', 4: 'v2', 5: 'v3'}
        
        lh_file = os.path.join("../nsddatapaper/mainfigures/SCIENCE.RSA", 'lh.highlevelvisual.mgz')
        rh_file = os.path.join("../nsddatapaper/mainfigures/SCIENCE.RSA", 'rh.highlevelvisual.mgz')

        # load the lh mask
        maskdata_lh = nib.load(lh_file).get_fdata().squeeze()
        maskdata_rh = nib.load(rh_file).get_fdata().squeeze()

        maskdata = np.hstack((maskdata_lh, maskdata_rh))

        for roi in range(1, 6):
            mask_name = ROIS[roi]

            #if not os.path.exists(rdm_file):

            # logical array of mask vertices
            vs_mask = maskdata == roi
            print(f'working on ROI: {mask_name}')

            masked_betas = betas[vs_mask, :]

            good_vox = [
                True if np.sum(
                    np.isnan(x)
                    ) == 0 else False for x in masked_betas]

            if np.sum(good_vox) != len(good_vox):
                print(f'found some NaN for ROI: {mask_name} - {sub}')

            masked_betas = masked_betas[good_vox, :]

            ### average for each category
            avg_betas_file = os.path.join(
                    betas_dir, f'{sub}_{mask_name}_betas_list_{targetspace}_category_averaged.npy'
            )
            
            avg_betas = np.matmul(masked_betas, category_mat) # shape of ([10000, 80])
            np.save(avg_betas_file, avg_betas)
            
            avg_betas = np.load(avg_betas_file)
            X = avg_betas.T
            #X /= np.sum(np.abs(X), axis=1).reshape(-1, 1) #normalize

            rdm = distance.cdist(X, X, metric='correlation')
            rdm = np.nan_to_num(rdm) # replace nan with zeros

            print(f'saving RDM for {mask_name}')
            rdm_file = os.path.join(
                        betas_dir, f'{sub}_{mask_name}_rdm_category_averaged.npy'
                )
            np.save(rdm_file, rdm)

# %%
