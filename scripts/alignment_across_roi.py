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
import logging

from src.utils import sample_participants, split_lists, show_matrix
from GW_methods.src.align_representations import Representation, AlignRepresentations, OptimizationConfig, VisualizationConfig
from GW_methods.src.utils.utils_functions import get_category_data, sort_matrix_with_categories

#%%
# Configure logging
logging.basicConfig(filename='alignment_across_roi.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

#%%
### Check carefully before running
# raise alert

compute_OT = False

n_subj = 8
n_groups = 2
subj_list = [f"subj0{i+1}" for i in range(8)]

roi_list = ['pVTC', 'aVTC', 'v1', 'v2', 'v3', 'OPA', 'PPA', 'RSC', 'MTL'] #['pVTC', 'aVTC', 'v1', 'v2', 'v3']
roi_list_1 = ['pVTC', 'aVTC', 'v1', 'v2', 'v3'] # cuda:3
roi_list_2 = ['OPA', 'PPA', 'RSC', 'MTL'] # cuda:3

# set the combination of roi pairs between roi_list_1 and roi_list_2
pairs_list = [(roi1, roi2) for roi1 in roi_list_1 for roi2 in roi_list_2]

# device = "cuda:3"
# roi_pairs = list(itertools.combinations(roi_list, 2))
# delete_results = False
# RDM_concat = False

# # across_concat
# device = "cuda:1"
# roi_list = roi_list_1
# roi_pairs = list(itertools.combinations(roi_list, 2))
# delete_results = True
# RDM_concat = True

# across_concat_2
# device = "cuda:2"
# roi_list = roi_list_2
# roi_pairs = list(itertools.combinations(roi_list, 2))
# delete_results = True
# RDM_concat = True

# across_concat_3
device = "cuda:3"
roi_list = roi_list_1 + roi_list_2
roi_pairs = pairs_list
delete_results = True
RDM_concat = True


# across
# device = "cuda:1"
# roi_list = roi_list_2
# roi_pairs = list(itertools.combinations(roi_list, 2))
# delete_results = True
# RDM_concat = False

# across_2
# device = "cuda:1"
# roi_list = roi_list_1 + roi_list_2
# roi_pairs = pairs_list
# delete_results = False
# RDM_concat = False

#%%

if delete_results:
    conform = input("Are you sure you want to delete the results? (y/n)")
    if conform != 'y':
        raise ValueError("Results are not deleted.")

#roi_list = ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]
n_sample = 10
seed_list = range(n_sample)
#seed_list = range(5, 10)

# log
logging.info(f"Start alignment with RDM concat: {RDM_concat}, compute OT: {compute_OT}, device: {device}")


#%%
# subjects groups for each seed
groups_list = []
for seed in seed_list:
    subj_list = sample_participants(n_subj, n_subj, seed)
    groups = split_lists(subj_list, n_groups)
    groups_list.append(groups)

# category data
category_mat = pd.read_csv("../data/category_mat_shared515.csv", index_col=0)
object_labels, category_idx_list, category_num_list, new_category_name_list = get_category_data(category_mat)
    
#%%
for seed_id, groups in enumerate(groups_list):
    seed = seed_list[seed_id]
    
    logging.info(f'Starting iteration for seed {seed}')
    
    ### set dataframes
    top_k_list = [1, 3, 5]
    top_k_accuracy = pd.DataFrame()
    #top_k_accuracy["top_n"] = top_k_list
    k_nearest_accuracy = pd.DataFrame()
    
    cat_accuracy = pd.DataFrame()
    #cat_accuracy["top_n"] = top_k_list
    
    df_rsa = pd.DataFrame()
    df_gwd = pd.DataFrame()
    
    for roi_pair in roi_pairs:
        roi1, roi2 = roi_pair
        logging.info(f'Processing ROI pair: {roi1}, {roi2}')
        representations = []
        for j, roi in enumerate(roi_pair):
            
            if RDM_concat:
                mean_RDM = np.load(f"/mnt/NAS/common_data/natural-scenes-dataset/rsa/roi_analyses/seed{seed}_group{j}_{roi}_zscored_fullrdm_shared515_correlation.npy")
            else:
                RDMs = []
                if n_groups == 1:
                    group = groups[0]
                else:
                    group = groups[j]
                
                for i in group:
                    RDM = np.load(f"/mnt/NAS/common_data/natural-scenes-dataset/rsa/roi_analyses/subj0{i+1}_{roi}_fullrdm_shared515_correlation.npy")
                    RDMs.append(RDM)
                RDMs = np.stack(RDMs)
                mean_RDM = np.mean(RDMs, axis=0)

            representation = Representation(
                name=f"{roi}",
                sim_mat=mean_RDM,
                metric="euclidean",
                get_embedding=True,
                MDS_dim=10,
                object_labels=object_labels,
                category_name_list=new_category_name_list,
                num_category_list=category_num_list,
                category_idx_list=category_idx_list,
                func_for_sort_sim_mat=sort_matrix_with_categories
            )
            representations.append(representation)

        
        main_results_dir = "../results/gw_alignment/"
        # main_results_dir = "/mnt/NAS/user_data/ken-takeda/GWOT/Takeda_NSD/gw_alignment"
        os.makedirs(main_results_dir, exist_ok=True)
        init_mat_plan = 'random'
        
        concat_or_not = '_concat' if RDM_concat else ''
        data_name = f"NSD_across_roi_seed{seed}{concat_or_not}"
        
        if n_groups == 1:
            data_name += "_single_group"

            
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
            multi_gpu=[0, 1, 2, 3]
        )
        
        alignment = AlignRepresentations(
            config=opt_config, 
            representations_list=representations,
            metric="euclidean",
            main_results_dir=main_results_dir,
            data_name=data_name, 
            # pairs_computed=pairs_computed,
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

        if n_groups == 1:
            fig_dir = f"../results/figs/across_roi/single_group/"
        else:
            fig_dir = f"../results/figs/across_roi/seed{seed}{concat_or_not}/"
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
        
        logging.info(f'RSA correlation for {roi1}, {roi2}: {alignment.RSA_corr}')

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
            delete_results=delete_results,
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
        
        logging.info(f'GWD for {roi1}, {roi2}: {gwds}')

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
        
        alignment.calc_accuracy(
            top_k_list=top_k_list, 
            eval_type="k_nearest",
            )

        alignment.plot_accuracy(
            eval_type="ot_plan", 
            fig_dir=fig_dir, 
            fig_name="accuracy_ot_plan.png"
            )
        
        top_k_accuracy = pd.concat([top_k_accuracy, alignment.top_k_accuracy], axis=1)
        k_nearest_accuracy = pd.concat([k_nearest_accuracy, alignment.k_nearest_matching_rate], axis=1)
        
        logging.info(f'Top-k accuracy for {roi1}, {roi2}: {alignment.top_k_accuracy}')
        logging.info(f'K-nearest accuracy for {roi1}, {roi2}: {alignment.k_nearest_matching_rate}')
        
        # category level
        eval_mat = np.matmul(category_mat.values, category_mat.values.T)
        alignment.calc_accuracy(
            top_k_list=top_k_list, 
            eval_type="ot_plan",
            #ot_to_evaluate=OT_sorted[0],
            eval_mat = eval_mat
        )
        
        alignment.plot_accuracy(
            eval_type="ot_plan", 
            fig_dir=fig_dir, 
            fig_name="category_level_accuracy_ot_plan.png"
            )
        
        cat_accuracy = pd.concat([cat_accuracy, alignment.top_k_accuracy], axis=1)
        
        logging.info(f'Category accuracy for {roi1}, {roi2}: {alignment.top_k_accuracy}')
        
        #top_k_accuracy = top_k_accuracy.T
        #top_k_accuracy.index.name = 'pair_name'
        #top_k_accuracy_all = pd.concat([top_k_accuracy_all, top_k_accuracy], axis=0)
        
        #cat_accuracy = cat_accuracy.T
        #cat_accuracy.index.name = 'pair_name'
        #cat_accuracy_all = pd.concat([cat_accuracy_all, cat_accuracy], axis=0)
        
    # save data
    if n_groups == 1:
        save_dir = f'../results/gw_alignment/across_roi/single_group/'
    else:
        save_dir = f'../results/gw_alignment/across_roi{concat_or_not}/seed{seed}/'
    os.makedirs(save_dir, exist_ok=True)
    
    top_k_accuracy = top_k_accuracy.T
    top_k_accuracy.index.name = 'pair_name'
    top_k_accuracy.to_csv(os.path.join(save_dir, 'top_k_accuracy.csv'))
    
    k_nearest_accuracy = k_nearest_accuracy.T
    k_nearest_accuracy.index.name = 'pair_name'
    k_nearest_accuracy.to_csv(os.path.join(save_dir, 'k_nearest_accuracy.csv'))
    
    cat_accuracy = cat_accuracy.T
    cat_accuracy.index.name = 'pair_name'
    cat_accuracy.to_csv(os.path.join(save_dir, 'category_accuracy.csv'))
    
    df_rsa = df_rsa.T
    df_rsa.index.name = 'pair_name'
    df_rsa.to_csv(os.path.join(save_dir, 'rsa_correlation.csv'))
    
    df_gwd = df_gwd.T
    df_gwd.index.name = 'pair_name'
    df_gwd.to_csv(os.path.join(save_dir, 'gw_distance.csv'))
    
    logging.info(f'Saved results for seed {seed}')
    
#%%
# concatenate results
top_k_accuracy_all = pd.DataFrame()
k_nearest_accuracy_all = pd.DataFrame()
cat_accuracy_all = pd.DataFrame()
rsa_corr_all = pd.DataFrame()
gwd_all = pd.DataFrame()

for seed in range(n_sample):
    if n_groups == 1:
        main_results_dir = f'../results/gw_alignment/across_roi/single_group/'
    else:
        main_results_dir = f'../results/gw_alignment/across_roi/seed{seed}/'
    
    top_k_accuracy = pd.read_csv(os.path.join(main_results_dir, 'top_k_accuracy.csv'))
    k_nearest_accuracy = pd.read_csv(os.path.join(main_results_dir, 'k_nearest_accuracy.csv'))
    cat_accuracy = pd.read_csv(os.path.join(main_results_dir, 'category_accuracy.csv'))
    rsa_corr = pd.read_csv(os.path.join(main_results_dir, 'rsa_correlation.csv'))
    gwd = pd.read_csv(os.path.join(main_results_dir, 'gw_distance.csv'))
    
    top_k_accuracy_all = pd.concat([top_k_accuracy_all, top_k_accuracy], axis=0)
    k_nearest_accuracy_all = pd.concat([k_nearest_accuracy_all, k_nearest_accuracy], axis=0)
    cat_accuracy_all = pd.concat([cat_accuracy_all, cat_accuracy], axis=0)
    rsa_corr_all = pd.concat([rsa_corr_all, rsa_corr], axis=0)
    gwd_all = pd.concat([gwd_all, gwd], axis=0)

# save
if n_groups == 1:
    save_dir = f'../results/gw_alignment/across_roi/single_group/'
else:
    save_dir = f'../results/gw_alignment/across_roi/'
os.makedirs(save_dir, exist_ok=True)
top_k_accuracy_all.to_csv(os.path.join(save_dir, 'top_k_accuracy.csv'))
k_nearest_accuracy_all.to_csv(os.path.join(save_dir, 'k_nearest_accuracy.csv'))
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
