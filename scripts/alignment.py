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
n_subj = 8
n_groups = 2
subj_list = [f"subj0{i+1}" for i in range(8)]
roi_list = ["pVTC"] #['pVTC', 'aVTC', 'v1', 'v2', 'v3']
n_sample =1

compute_OT = False

#%%
def sample_participants(N, Z, seed):
    random.seed(seed)
    participants_list = random.sample(range(N), Z)
    return participants_list

def split_lists(list, n_groups): 
    N = len(list) // n_groups
    result = []
    for i in range(n_groups):
        list_i = list[N*i : N*(i+1)]
        result.append(list_i)
    return result

groups_list = []
for seed in range(n_sample):
    subj_list = sample_participants(n_subj, n_subj, seed)
    groups = split_lists(subj_list, n_groups)
    groups_list.append(groups)
    
# category data
category_mat = pd.read_csv("../data/category_mat_shared515.csv")
object_labels, category_idx_list, category_num_list, new_category_name_list = get_category_data(category_mat)
    
#%%
for roi in roi_list:
    for seed, groups in enumerate(groups_list):
        
        representations = []
        for j, group in enumerate(groups):
            RDMs = []
            for i in group:
                RDM = np.load(f"/home1/data/common-data/natural-scenes-dataset/rsa/roi_analyses/subj0{i+1}_{roi}_fullrdm_shared515_correlation.npy")
                RDMs.append(RDM)
            RDMs = np.stack(RDMs)
            mean_RDM = np.mean(RDMs, axis=0)

            representation = Representation(
                name=f"Group{j+1}_{roi}",
                sim_mat=mean_RDM,
                metric="euclidean",
                get_embedding=False,
                object_labels=object_labels,
                category_name_list=new_category_name_list,
                num_category_list=category_num_list,
                category_idx_list=category_idx_list,
                func_for_sort_sim_mat=sort_matrix_with_categories
            )
            representations.append(representation)

        opt_config = OptimizationConfig(
            init_mat_plan="random",
            db_params={"drivername": "sqlite"},
            num_trial=100,
            n_iter=1, 
            max_iter=200,
            sampler_name="tpe", 
            eps_list=[1e-4, 1e-2],
            eps_log=True,
            )

        alignment = AlignRepresentations(
            config=opt_config, 
            representations_list=representations,
            metric="euclidean",
            main_results_dir="../results/gw_alignment/",
            data_name=f"NSD_within_{roi}_seed{seed}", 
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

        os.makedirs(f"../results/figs/{roi}/seed{seed}/", exist_ok=True)

        alignment.show_sim_mat(
            sim_mat_format="sorted",
            visualization_config=vis_config, 
            show_distribution=False,
            fig_dir=f"../results/figs/{roi}/seed{seed}/"
            )
        alignment.RSA_get_corr()

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

        alignment.gw_alignment(
            compute_OT=compute_OT,
            delete_results=False,
            OT_format="sorted",
            visualization_config=vis_config_OT,
            fig_dir=f"../results/figs/{roi}/seed{seed}/"
            )

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
            fig_dir=f"../results/figs/{roi}/seed{seed}/"
            )

        ## Calculate the accuracy of the optimized OT matrix
        top_k_list = [1, 3, 5]

        alignment.calc_accuracy(
            top_k_list=top_k_list, 
            eval_type="ot_plan"
            )

        alignment.plot_accuracy(
            eval_type="ot_plan", 
            fig_dir=f"../results/figs/{roi}/seed{seed}/", 
            fig_name="accuracy_ot_plan.png"
            )
    
# %%
