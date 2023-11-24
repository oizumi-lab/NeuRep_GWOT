#%%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../'))

import numpy as np
import pandas as pd
import pickle as pkl
import torch
import random
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils import sample_participants, split_lists, get_meta_RDM, sum_of_block_matrices
from GW_methods.src.align_representations import Representation, AlignRepresentations, OptimizationConfig, VisualizationConfig
from GW_methods.src.utils.utils_functions import get_category_data, sort_matrix_with_categories

#%%
n_subj = 8
n_groups = 2
subj_list = [f"subj0{i+1}" for i in range(8)]
roi_list = ['pVTC', 'aVTC', 'v1', 'v2', 'v3'] #['pVTC', 'aVTC', 'v1', 'v2', 'v3']
n_sample = 1

compute_OT = False

groups_list = []
for seed in range(n_sample):
    subj_list = sample_participants(n_subj, n_subj, seed)
    groups = split_lists(subj_list, n_groups)
    groups_list.append(groups)
    
# get RDMs
seed = 0
groups = groups_list[seed]

meta_RDM_list = []
for group in groups:
    RDM_list = [] # container of mean RDMs of all roi
    for roi in roi_list:
        RDMs = []
        for i in group:
            RDM = np.load(f"/home1/data/common-data/natural-scenes-dataset/rsa/roi_analyses/subj0{i+1}_{roi}_fullrdm_shared515_correlation.npy")
            RDMs.append(RDM)
        RDMs = np.stack(RDMs)
        mean_RDM = np.mean(RDMs, axis=0) # group averaged RDM for 515 stimuls
        RDM_list.append(mean_RDM)

    meta_RDM = get_meta_RDM(RDM_list, metric='correlation')
    meta_RDM_list.append(meta_RDM)
    
#%%
### 
representations = []
for i, meta_RDM in enumerate(meta_RDM_list):
    representation = Representation(
        name=f"Group{i+1}_seed{seed}",
        sim_mat=meta_RDM,
        metric="euclidean",
        get_embedding=False,
        #object_labels=object_labels,
        #category_name_list=new_category_name_list,
        #num_category_list=category_num_list,
        #category_idx_list=category_idx_list,
        #func_for_sort_sim_mat=sort_matrix_with_categories
    )
    representations.append(representation)


main_results_dir = "../results/gw_alignment/"
init_mat_plan = 'random'
data_name = f"NSD_concat_roi_seed{seed}"

device = 'cuda'

if device == "cuda":
    sinkhorn_method = "sinkhorn_log" # please choose the method of sinkhorn implemented by POT (URL : https://pythonot.github.io/gen_modules/ot.bregman.html#id87). For using GPU, "sinkhorn_log" is recommended.
    data_type= "float"
    to_types='torch'
    multi_gpu = True # This parameter is only designed for "torch". # "True" : all the GPU installed in your environment are used, "list (e.g.[0,2,3])"" : cuda:0,2,3, and "False" : single gpu will be used.

elif device == "cpu":
    sinkhorn_method = "sinkhorn"
    data_type = "double"
    to_types = 'numpy'
    multi_gpu = False
    
opt_config = OptimizationConfig(
    init_mat_plan=init_mat_plan,
    db_params={"drivername": "sqlite"},
    num_trial=100,
    n_iter=1, 
    max_iter=1000,
    sampler_name="tpe", 
    eps_list=[1e-4, 1e-2],
    eps_log=True,
    device=device,
    to_types=to_types,
    data_type=data_type,
    sinkhorn_method=sinkhorn_method,
    multi_gpu=multi_gpu
    
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
    #cbar_ticks_size=30,
    #font = "Arial",
    #cbar_label="Dissimilarity",
    #cbar_label_size=40,
    )

fig_dir = f"../results/figs/concat_roi/seed{seed}/"

os.makedirs(fig_dir, exist_ok=True)

alignment.show_sim_mat(
    sim_mat_format="default",
    visualization_config=vis_config, 
    show_distribution=False,
    fig_dir=fig_dir
    )

alignment.RSA_get_corr()
#rsa_corr = pd.DataFrame([alignment.RSA_corr], index=['correlation'])
#df_rsa = pd.concat([df_rsa, rsa_corr], axis=1)

#%%
vis_config_OT = VisualizationConfig(
    figsize=(8, 6), 
    #title_size = 15, 
    cmap = "rocket_r",
    #cbar_ticks_size=30,
    #font = "Arial",
    #cbar_label="Probability",
    #cbar_label_size=40,
    #color_labels = new_color_order,
    #color_label_width = 5,
    #xlabel=f"93 colors of {name_list[0]}",
    #xlabel_size = 40,
    #ylabel=f"93 colors of {name_list[1]}",
    #ylabel_size = 40,
    )

OT_default = alignment.gw_alignment(
    compute_OT=compute_OT,
    delete_results=False,
    OT_format="default",
    return_data=True,
    visualization_config=vis_config_OT,
    fig_dir=fig_dir,
    save_dataframe=True
    )

# record gwd
#gwds = {}
#for pairwise in alignment.pairwise_list:
#    pair_name = pairwise.pair_name
#    
#    study_name = data_name + '_' + pair_name
#    df = pd.read_csv(os.path.join(main_results_dir, study_name, init_mat_plan, study_name+'.csv'))
#    gwds[pair_name] = df['value'].min()
#gwds = pd.DataFrame([gwds], index=['gwd'])
#df_gwd = pd.concat([df_gwd, gwds], axis=1)

vis_config_log = VisualizationConfig(
    figsize=(8, 6), 
    #title_size = 15, 
    #cbar_ticks_size=30,
    #font = "Arial",
    #fig_ext="svg",
    #xlabel_size=35,
    #xticks_size=30,
    #xticks_rotation=0,
    #ylabel_size=35,
    #yticks_size=30,
    #cbar_label_size=30,
    plot_eps_log=True
    )

alignment.show_optimization_log(
    visualization_config=vis_config_log, 
    fig_dir=fig_dir
    )

## Calculate the accuracy of the optimized OT matrix
top_k_list = [1, 3, 5]

alignment.calc_accuracy(
    top_k_list=top_k_list, 
    eval_type="ot_plan"
    )

alignment.plot_accuracy(
    eval_type="ot_plan", 
    fig_dir=fig_dir, 
    fig_name="accuracy_ot_plan.png"
    )

#top_k_accuracy = pd.concat([top_k_accuracy, alignment.top_k_accuracy])

# category level
#eval_mat = np.matmul(category_mat.values, category_mat.values.T)
#alignment.calc_accuracy(
#    top_k_list=top_k_list, 
#    eval_type="ot_plan",
#    ot_to_evaluate=OT_sorted[0],
#    eval_mat = eval_mat
#)
#
#alignment.plot_accuracy(
#    eval_type="ot_plan", 
#    fig_dir=fig_dir, 
#    fig_name="category_level_accuracy_ot_plan.png"
#    )
#
#cat_accuracy = pd.concat([cat_accuracy, alignment.top_k_accuracy])
        
#    # save data
#    save_dir = f'../results/gw_alignment/within{roi}/'
#    os.makedirs(save_dir, exist_ok=True)
#
#    top_k_accuracy.to_csv(os.path.join(save_dir, 'top_k_accuracy.csv'))
#    cat_accuracy.to_csv(os.path.join(save_dir, 'category_accuracy.csv'))
#    
#    df_rsa = df_rsa.T
#    df_rsa.index.name = 'pair_name'
#    df_rsa.to_csv(os.path.join(save_dir, 'rsa_correlation.csv'))
#    
#    df_gwd = df_gwd.T
#    df_gwd.index.name = 'pair_name'
#    df_gwd.to_csv(os.path.join(save_dir, 'gw_distance.csv'))
    
# %%
### show the result
block_size = 515

OT = OT_default[0] * len(roi_list)

sum_block_mat = sum_of_block_matrices(OT, block_size)
ticks = np.array(range(len(roi_list))) + 0.5

plt.figure()
sns.heatmap(sum_block_mat, annot=True, cmap='rocket', fmt=".2f")
# set the labels of each axis with roi list
plt.xticks(ticks, roi_list, fontsize=15)
plt.yticks(ticks, roi_list, fontsize=15)

plt.show()
plt.savefig(os.path.join(fig_dir, 'OT_concat_roi.png'), dpi=300)


# %%
