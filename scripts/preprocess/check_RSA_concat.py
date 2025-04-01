#%%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../'))
sys.path.append(os.path.join(os.getcwd(), '../../'))

import numpy as np
import pandas as pd
import pickle as pkl
import torch
import random
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from src.utils import sample_participants, split_lists
from GW_methods.src.align_representations import Representation, AlignRepresentations, OptimizationConfig, VisualizationConfig
from GW_methods.src.utils.utils_functions import get_category_data, sort_matrix_with_categories


#%%

n_subj = 8
n_groups = 2
subj_list = [f"subj0{i+1}" for i in range(8)]

roi_list = ['v1', 'v2', 'v3', 'pVTC', 'aVTC', 'OPA', 'PPA', 'RSC', 'MTL'] #, 'OPA', 'PPA', 'RSC', 'MTL'


# roi_list = ['pVTC']
n_sample = 10
seed_list = range(n_sample)
#seed_list = range(5, 10)
#seed_list = [4]

one_vs_one = False
if one_vs_one:
    seed_list = [0]

# device = 'cuda:2'
get_embedding = False

#%%
groups_list = []
for seed in seed_list:
    if one_vs_one:
        subj_pairs = itertools.combinations(range(len(subj_list)), 2)
        groups = [[[pair[0]], [pair[1]]] for pair in subj_pairs]
        groups_list = groups
    else:
        subj_list = sample_participants(n_subj, n_subj, seed)
        groups = split_lists(subj_list, n_groups)
        groups_list.append(groups)
    
# category data
category_mat = pd.read_csv("../../data/category_mat_shared515.csv", index_col=0)
object_labels, category_idx_list, category_num_list, new_category_name_list = get_category_data(category_mat)


condition_names = ['RDM_mean', 'concat', 'concat_zscored']
#%%
for roi in roi_list:
    df_rsa = pd.DataFrame()
    
    for condition_name in condition_names:
        
        for seed_id, groups in enumerate(groups_list):
            seed = seed_list[seed_id]
            
            representations = []
            for j, group in enumerate(groups):
                if  condition_name == 'concat' or condition_name == 'concat_zscored':
                    normalize_str = 'zscored_' if condition_name == 'concat_zscored' else ''
                    mean_RDM = np.load(f"/mnt/NAS/common_data/natural-scenes-dataset/rsa/roi_analyses/seed{seed}_group{j}_{roi}_{normalize_str}fullrdm_shared515_correlation.npy")
                else:
                    RDMs = []
                    for i in group:
                        RDM = np.load(f"/mnt/NAS/common_data/natural-scenes-dataset/rsa/roi_analyses/subj0{i+1}_{roi}_fullrdm_shared515_correlation.npy")
                        RDMs.append(RDM)
                    RDMs = np.stack(RDMs)
                    mean_RDM = np.mean(RDMs, axis=0)

                representation = Representation(
                    name=f"Group{j+1}_{roi}",
                    sim_mat=mean_RDM,
                    metric="euclidean",
                    get_embedding=get_embedding,
                    MDS_dim=50,
                    object_labels=object_labels,
                    category_name_list=new_category_name_list,
                    num_category_list=category_num_list,
                    category_idx_list=category_idx_list,
                    func_for_sort_sim_mat=sort_matrix_with_categories
                )
                representations.append(representation)

            device = 'cuda:2'
            init_mat_plan = 'random'
            main_results_dir = "/mnt/NAS/user_data/ken-takeda/GWOT/Takeda_NSD/gw_alignment"
            data_name = f"NSD_within_check_RSA_{roi}_{condition_name}_seed{seed}"
            
            opt_config = OptimizationConfig(
                init_mat_plan=init_mat_plan,
                db_params={"drivername": "sqlite"},
                num_trial=100,
                n_iter=1, 
                max_iter=200,
                sampler_name="tpe", 
                eps_list=[1e-4, 1e-2],
                eps_log=True,
                device=device,
                to_types='torch',
                multi_gpu=False
            )
            
            alignment = AlignRepresentations(
                config=opt_config, 
                representations_list=representations,
                metric="euclidean",
                main_results_dir=main_results_dir,
                data_name=data_name, 
                )

            vis_config = VisualizationConfig(
                figsize=(8, 6), 
                #title_size = 15, 
                cmap = "rocket_r",
                cbar_ticks_size=30,
                font = "Arial",
                cbar_label="Dissimilarity",
                cbar_label_size=40,
                xlabel=f"515 images",
                xlabel_size = 35,
                ylabel=f"515 images",
                ylabel_size = 35,
                )
            
            alignment.RSA_get_corr()
            corr = alignment.RSA_corr.values()
            rsa_corr = pd.DataFrame({"correlation": corr, 
                                     'condition': condition_name})
            df_rsa = pd.concat([df_rsa, rsa_corr], axis=0)
                
        
    # plot RSA
    plt.figure(figsize=(10, 6))
    sns.swarmplot(data=df_rsa, x='condition', y='correlation')
    plt.title(f"RSA correlation {condition_name} \n {roi}")
    plt.show()
# %%

# cross conditional RSA
# compare RDM_mean and concat_zscored
df_rsa = pd.DataFrame()
for roi in roi_list:
    for seed_id, groups in enumerate(groups_list):
        seed = seed_list[seed_id]
        
        for j in range(2):
            representations = []
            for condition_name in ['RDM_mean', 'concat_zscored']:
            # for j, group in enumerate(groups):
                # j = 0
                group = groups[j]
                if  condition_name == 'concat' or condition_name == 'concat_zscored':
                    normalize_str = 'zscored_' if condition_name == 'concat_zscored' else ''
                    mean_RDM = np.load(f"/mnt/NAS/common_data/natural-scenes-dataset/rsa/roi_analyses/seed{seed}_group{j}_{roi}_{normalize_str}fullrdm_shared515_correlation.npy")
                else:
                    RDMs = []
                    for i in group:
                        RDM = np.load(f"/mnt/NAS/common_data/natural-scenes-dataset/rsa/roi_analyses/subj0{i+1}_{roi}_fullrdm_shared515_correlation.npy")
                        RDMs.append(RDM)
                    RDMs = np.stack(RDMs)
                    mean_RDM = np.mean(RDMs, axis=0)

                representation = Representation(
                    name=f"Group{j+1}_{roi}",
                    sim_mat=mean_RDM,
                    metric="euclidean",
                    get_embedding=get_embedding,
                    MDS_dim=50,
                    object_labels=object_labels,
                    category_name_list=new_category_name_list,
                    num_category_list=category_num_list,
                    category_idx_list=category_idx_list,
                    func_for_sort_sim_mat=sort_matrix_with_categories
                )
                representations.append(representation)

            device = 'cuda:2'
            init_mat_plan = 'random'
            main_results_dir = "/mnt/NAS/user_data/ken-takeda/GWOT/Takeda_NSD/gw_alignment"
            data_name = f"NSD_within_check_RSA_{roi}_{condition_name}_seed{seed}"
            
            opt_config = OptimizationConfig(
                init_mat_plan=init_mat_plan,
                db_params={"drivername": "sqlite"},
                num_trial=100,
                n_iter=1, 
                max_iter=200,
                sampler_name="tpe", 
                eps_list=[1e-4, 1e-2],
                eps_log=True,
                device=device,
                to_types='torch',
                multi_gpu=False
            )
            
            alignment = AlignRepresentations(
                config=opt_config, 
                representations_list=representations,
                metric="euclidean",
                main_results_dir=main_results_dir,
                data_name=data_name, 
                )

            vis_config = VisualizationConfig(
                figsize=(8, 6), 
                #title_size = 15, 
                cmap = "rocket_r",
                cbar_ticks_size=30,
                font = "Arial",
                cbar_label="Dissimilarity",
                cbar_label_size=40,
                xlabel=f"515 images",
                xlabel_size = 35,
                ylabel=f"515 images",
                ylabel_size = 35,
                )
            
            alignment.RSA_get_corr()
            corr = alignment.RSA_corr.values()
            rsa_corr = pd.DataFrame({"correlation": corr, 
                                    'roi': roi})
            
            df_rsa = pd.concat([df_rsa, rsa_corr], axis=0)
    
# plot RSA
plt.figure(figsize=(10, 6))
sns.swarmplot(data=df_rsa, x='roi', y='correlation')
# set lim
plt.ylim([0.9, 1])
plt.title(f"RSA correlation {condition_name}")
plt.show()
    
# %%
