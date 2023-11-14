#%%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import re
import numpy as np
import torch
from src.load_img import GWD_Dataset
#%%
save_path = '../../data/models/shared515'
dataset_path = '/home1/data/common-data/natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/shared_515'


model_list = [
    'AlexNet', 
    'VGG19',
    'CLIP_B16_OpenAI',
    'CLIP_B16_datacomp_l_s1b_b8k',
    'CLIP_B16_datacomp_xl_s13b_b90k',
    'CLIP_B16_laion2B-s34B-b88K', 
    # 'CLIP_L14_commonpool_xl_laion_s13b_b90k', 
    'ViT_B16_ImageNet1K', 
    'ViT_B16_ImageNet21K',
]

for model_name in model_list:
    test = GWD_Dataset(model_name=model_name, save_path=save_path, dataset_path=dataset_path)
    cosine_dis_sim, label = test.extract()

#%%
### as label is not in the order of image numbers, I need to sort indices 
def sort_2d_array(arr, sort_indices_x, sort_indices_y):
    """
    Sorts a 2D array based on the provided sorting indices for each axis.

    Parameters:
    arr (2D array): The 2D array to be sorted.
    sort_indices_x (list of ints): The sorting indices for the x-axis (rows).
    sort_indices_y (list of ints): The sorting indices for the y-axis (columns).

    Returns:
    2D array: The sorted array.
    """

    # Sort rows
    sorted_arr = arr[sort_indices_x]

    # Sort columns
    sorted_arr = sorted_arr[:, sort_indices_y]

    return sorted_arr

#%%
classes = []
for i in label:
    classes.append(test.dataset.classes[i])

# %%
sorted_list = sorted(classes, key=lambda s: int(re.search(r'\d+', s).group()))
index_list = [classes.index(element) for element in sorted_list]

index_list = np.array(index_list)
np.save('../../data/models/shared515/index_list', index_list)
# %%
### load the saved RDMs and sort them 
for model in model_list:
    sim_mat = torch.load(os.path.join(save_path, 'sim_mat', f'{model.lower()}_conv.pt'))
    sorted_sim_mat = sort_2d_array(sim_mat, index_list, index_list)
    torch.save(sorted_sim_mat, os.path.join(save_path, 'sim_mat', f'sorted_{model.lower()}_conv.pt'))
# %%
