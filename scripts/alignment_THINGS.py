#%%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../'))

import numpy as np
import pandas as pd
import pickle as pkl
import torch
import random

from GW_methods.src.align_representations import Representation, AlignRepresentations, OptimizationConfig, VisualizationConfig
from GW_methods.src.utils.utils_functions import get_category_data, sort_matrix_with_categories

#%%
compute_OT = True

n_subj = 8
roi_list = ['aVTC'] #, 'pVTC', 'v3', 'v2', 'v1'

# category data
nsd_category_mat = pd.read_csv("../data/category_mat_shared515.csv", index_col=0)
nsd_object_labels, nsd_category_idx_list, nsd_category_num_list, nsd_new_category_name_list = get_category_data(nsd_category_mat)

things_category_mat = pd.read_csv("../data/THINGS/category_mat_manual_preprocessed.csv", index_col=0)
things_object_labels, things_category_idx_list, things_category_num_list, things_new_category_name_list = get_category_data(things_category_mat)

#%%
# fMRI
for roi in roi_list:
    RDMs = []
    for i in range(n_subj):
        RDM = np.load(f"/home1/data/common-data/natural-scenes-dataset/rsa/roi_analyses/subj0{i+1}_{roi}_fullrdm_shared515_correlation.npy")
        RDMs.append(RDM)
    RDMs = np.stack(RDMs)
    mean_RDM = np.mean(RDMs, axis=0)
    
    fMRI = Representation(
        name=f"fMRI_{roi}",
        sim_mat=mean_RDM,
        get_embedding=False,
        object_labels=nsd_object_labels,
        category_name_list=nsd_new_category_name_list,
        num_category_list=nsd_category_num_list,
        category_idx_list=nsd_category_idx_list,
        func_for_sort_sim_mat=sort_matrix_with_categories
    )
    
    # behavior
    embedding = pd.read_csv('../data/THINGS/spose_embedding_66d_sorted.txt', sep='\t', header=None).values
    
    behav = Representation(
        name="behavior",
        embedding=embedding,
        metric="cosine",
        get_embedding=False,
        object_labels=things_object_labels,
        category_name_list=things_new_category_name_list,
        num_category_list=things_category_num_list,
        category_idx_list=things_category_idx_list,
        func_for_sort_sim_mat=sort_matrix_with_categories
    )
    
    representations = [fMRI, behav]
    
    main_results_dir = f"../results/gw_alignment/behavior/"
    init_mat_plan = 'random'
    data_name = f"NSD_{roi}_vs_THINGS_behavior"

    opt_config = OptimizationConfig(
        init_mat_plan=init_mat_plan,
        db_params={"drivername": "sqlite"},
        num_trial=100,
        n_iter=1, 
        max_iter=1000,
        sampler_name="tpe", 
        eps_list=[1e-5, 1e-1],#[1e-10, 1]
        eps_log=True,
        to_types='torch',
        device='cuda:0',
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
        #cbar_ticks_size=30,
        #font = "Arial",
        #cbar_label="Dissimilarity",
        #cbar_label_size=40,
        )
    
    fig_dir = f"../results/figs/{roi}/behavior/"
    os.makedirs(fig_dir, exist_ok=True)
    
    alignment.show_sim_mat(
        sim_mat_format="sorted",
        visualization_config=vis_config, 
        show_distribution=True,
        fig_dir=fig_dir
        )
    
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

    OT_sorted = alignment.gw_alignment(
        compute_OT=compute_OT,
        delete_results=False,
        OT_format="sorted",
        return_data=True,
        save_dataframe=True,
        visualization_config=vis_config_OT,
        fig_dir=fig_dir
        )
    
    
# %%
