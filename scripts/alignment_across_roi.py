#%%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../'))

import numpy as np
import pandas as pd
import pickle as pkl
import torch
import random
import itertools
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils import sample_participants, split_lists, show_matrix
from GW_methods.src.align_representations import Representation, AlignRepresentations, OptimizationConfig, VisualizationConfig
from GW_methods.src.utils.utils_functions import get_category_data, sort_matrix_with_categories

#%%
n_subj = 8
n_groups = 2
subj_list = [f"subj0{i+1}" for i in range(8)]
roi_list = ['pVTC', 'aVTC', 'v1', 'v2', 'v3'] #['pVTC', 'aVTC', 'v1', 'v2', 'v3']
n_sample = 5

compute_OT = False

#%%
# subjects groups for each seed
groups_list = []
for seed in range(n_sample):
    subj_list = sample_participants(n_subj, n_subj, seed)
    groups = split_lists(subj_list, n_groups)
    groups_list.append(groups)
    
# roi pairs
roi_pairs = list(itertools.combinations(roi_list, 2))

# category data
category_mat = pd.read_csv("../data/category_mat_shared515.csv", index_col=0)
object_labels, category_idx_list, category_num_list, new_category_name_list = get_category_data(category_mat)
    
#%%
for seed, groups in enumerate(groups_list):
    ### set dataframes
    top_k_list = [1, 3, 5]
    top_k_accuracy = pd.DataFrame()
    #top_k_accuracy["top_n"] = top_k_list
    
    cat_accuracy = pd.DataFrame()
    #cat_accuracy["top_n"] = top_k_list
    
    df_rsa = pd.DataFrame()
    df_gwd = pd.DataFrame()
    
    for roi_pair in roi_pairs:
        roi1, roi2 = roi_pair
        representations = []
        for j, group in enumerate(groups):
            roi = roi_pair[j]
            RDMs = []
            for i in group:
                RDM = np.load(f"/home1/data/common-data/natural-scenes-dataset/rsa/roi_analyses/subj0{i+1}_{roi}_fullrdm_shared515_correlation.npy")
                RDMs.append(RDM)
            RDMs = np.stack(RDMs)
            mean_RDM = np.mean(RDMs, axis=0)

            representation = Representation(
                name=f"{roi}",
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

        
        main_results_dir = "../results/gw_alignment/"
        init_mat_plan = 'random'
        data_name = f"NSD_across_roi_seed{seed}"
        
        device = 'cuda:1'

        if "cuda" in device:
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
            max_iter=500,
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

        fig_dir = f"../results/figs/across_roi/seed{seed}/"
        os.makedirs(fig_dir, exist_ok=True)

        alignment.show_sim_mat(
            sim_mat_format="sorted",
            visualization_config=vis_config, 
            show_distribution=False,
            fig_dir=fig_dir
            )
        
        alignment.RSA_get_corr()
        rsa_corr = pd.DataFrame([alignment.RSA_corr], index=['correlation'])
        df_rsa = pd.concat([df_rsa, rsa_corr], axis=1)

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
            visualization_config=vis_config_OT,
            fig_dir=fig_dir,
            save_dataframe=True
            )
        
        # record gwd
        gwds = {}
        for pairwise in alignment.pairwise_list:
            pair_name = pairwise.pair_name
            
            study_name = data_name + '_' + pair_name
            df = pd.read_csv(os.path.join(main_results_dir, study_name, init_mat_plan, study_name+'.csv'))
            gwds[pair_name] = df['value'].min()
        gwds = pd.DataFrame([gwds], index=['gwd'])
        df_gwd = pd.concat([df_gwd, gwds], axis=1)

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
        alignment.calc_accuracy(
            top_k_list=top_k_list, 
            eval_type="ot_plan"
            )

        alignment.plot_accuracy(
            eval_type="ot_plan", 
            fig_dir=fig_dir, 
            fig_name="accuracy_ot_plan.png"
            )
        
        top_k_accuracy = pd.concat([top_k_accuracy, alignment.top_k_accuracy], axis=1)
        
        # category level
        eval_mat = np.matmul(category_mat.values, category_mat.values.T)
        alignment.calc_accuracy(
            top_k_list=top_k_list, 
            eval_type="ot_plan",
            ot_to_evaluate=OT_sorted[0],
            eval_mat = eval_mat
        )
        
        alignment.plot_accuracy(
            eval_type="ot_plan", 
            fig_dir=fig_dir, 
            fig_name="category_level_accuracy_ot_plan.png"
            )
        
        cat_accuracy = pd.concat([cat_accuracy, alignment.top_k_accuracy], axis=1)
        
        #top_k_accuracy = top_k_accuracy.T
        #top_k_accuracy.index.name = 'pair_name'
        #top_k_accuracy_all = pd.concat([top_k_accuracy_all, top_k_accuracy], axis=0)
        
        #cat_accuracy = cat_accuracy.T
        #cat_accuracy.index.name = 'pair_name'
        #cat_accuracy_all = pd.concat([cat_accuracy_all, cat_accuracy], axis=0)
        
    # save data
    save_dir = f'../results/gw_alignment/across_roi/seed{seed}/'
    os.makedirs(save_dir, exist_ok=True)
    
    top_k_accuracy = top_k_accuracy.T
    top_k_accuracy.index.name = 'pair_name'
    top_k_accuracy.to_csv(os.path.join(save_dir, 'top_k_accuracy.csv'))
    
    cat_accuracy = cat_accuracy.T
    cat_accuracy.index.name = 'pair_name'
    cat_accuracy.to_csv(os.path.join(save_dir, 'category_accuracy.csv'))
    
    df_rsa = df_rsa.T
    df_rsa.index.name = 'pair_name'
    df_rsa.to_csv(os.path.join(save_dir, 'rsa_correlation.csv'))
    
    df_gwd = df_gwd.T
    df_gwd.index.name = 'pair_name'
    df_gwd.to_csv(os.path.join(save_dir, 'gw_distance.csv'))
    
#%%
# concatenate results
top_k_accuracy_all = pd.DataFrame()
cat_accuracy_all = pd.DataFrame()
rsa_corr_all = pd.DataFrame()
gwd_all = pd.DataFrame()

for seed in range(n_sample):
    main_results_dir = f'../results/gw_alignment/across_roi/seed{seed}/'
    
    top_k_accuracy = pd.read_csv(os.path.join(main_results_dir, 'top_k_accuracy.csv'))
    cat_accuracy = pd.read_csv(os.path.join(main_results_dir, 'category_accuracy.csv'))
    rsa_corr = pd.read_csv(os.path.join(main_results_dir, 'rsa_correlation.csv'))
    gwd = pd.read_csv(os.path.join(main_results_dir, 'gw_distance.csv'))
    
    top_k_accuracy_all = pd.concat([top_k_accuracy_all, top_k_accuracy], axis=0)
    cat_accuracy_all = pd.concat([cat_accuracy_all, cat_accuracy], axis=0)
    rsa_corr_all = pd.concat([rsa_corr_all, rsa_corr], axis=0)
    gwd_all = pd.concat([gwd_all, gwd], axis=0)

# save
save_dir = f'../results/gw_alignment/across_roi/'
os.makedirs(save_dir, exist_ok=True)
top_k_accuracy_all.to_csv(os.path.join(save_dir, 'top_k_accuracy.csv'))
cat_accuracy_all.to_csv(os.path.join(save_dir, 'category_accuracy.csv'))
rsa_corr_all.to_csv(os.path.join(save_dir, 'rsa_correlation.csv'))
gwd_all.to_csv(os.path.join(save_dir, 'gw_distance.csv'))

#%%
### show summary figure
    
#%%
#main_results_dir = f'../results/gw_alignment/across_roi/'
#fig_dir = f"../results/figs/across_roi/"
#
## top k acc
#df = pd.read_csv(os.path.join(main_results_dir, 'top_k_accuracy.csv'))
#top_k = 1
#title = f'top{top_k} matching rate'
#file_name = f'top_{top_k}_accuracy'
#
#show_matrix(df, col_name=str(top_k), title=title, save_dir=fig_dir, file_name=file_name, first_items=roi_list, second_items=roi_list)
## %%
## category
#df = pd.read_csv(os.path.join(main_results_dir, 'category_accuracy.csv')) 
#top_k = 1
#title = f'category level accuracy'
#file_name = f'category_top_{top_k}_accuracy'
#
#show_matrix(df, col_name=str(top_k), title=title, save_dir=fig_dir, file_name=file_name, first_items=roi_list, second_items=roi_list)
#
##%%
## rsa
#df = pd.read_csv(os.path.join(main_results_dir, 'rsa_correlation.csv'))
#col_name = 'correlation'
#title = 'RSA correlation'
#file_name = f'rsa_all_pairs'
#
#show_matrix(df, col_name=col_name, title=title, save_dir=fig_dir, file_name=file_name, first_items=roi_list, second_items=roi_list)
#
## %%
## gwd
#df = pd.read_csv(os.path.join(main_results_dir, 'gw_distance.csv'))
#col_name = 'gwd'
#title = 'GWD'
#file_name = f'gwd_all_pairs'
#
#show_matrix(df, col_name=col_name, title=title, save_dir=fig_dir, file_name=file_name, first_items=roi_list, second_items=roi_list)
# %%
