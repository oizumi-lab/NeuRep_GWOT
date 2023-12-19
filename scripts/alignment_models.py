#%%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../'))

import numpy as np
import pandas as pd
import pickle as pkl
import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import sample_participants, split_lists, get_meta_RDM, sum_of_block_matrices, show_matrix
from GW_methods.src.align_representations import Representation, AlignRepresentations, OptimizationConfig, VisualizationConfig
from GW_methods.src.utils.utils_functions import get_category_data, sort_matrix_with_categories

#%%
n_subj = 8
roi_list = ['pVTC', 'aVTC', 'v1', 'v2', 'v3'] #['pVTC', 'aVTC', 'v1', 'v2', 'v3']

compute_OT = True
z_transform = False
layer_wise = False

# category data
category_mat = pd.read_csv("../data/category_mat_shared515.csv", index_col=0)
object_labels, category_idx_list, category_num_list, new_category_name_list = get_category_data(category_mat)

#%%
# fMRI
top_k_accuracy_all = pd.DataFrame()
cat_accuracy_all = pd.DataFrame()
df_rsa_all = pd.DataFrame()
df_gwd_all = pd.DataFrame()

for roi in roi_list:
    ### set dataframes
    top_k_list = [1, 3, 5]
    top_k_accuracy = pd.DataFrame()
    #top_k_accuracy["top_n"] = top_k_list
    
    cat_accuracy = pd.DataFrame()
    #cat_accuracy["top_n"] = top_k_list
    
    df_rsa = pd.DataFrame()
    df_gwd = pd.DataFrame()
    
    RDMs = []
    for i in range(n_subj):
        RDM = np.load(f"/home1/data/common-data/natural-scenes-dataset/rsa/roi_analyses/subj0{i+1}_{roi}_fullrdm_shared515_correlation.npy")
        RDMs.append(RDM)
    RDMs = np.stack(RDMs)
    mean_RDM = np.mean(RDMs, axis=0)
    
    # Z-transform
    if z_transform:
        mean_RDM = (mean_RDM - mean_RDM.mean()) / mean_RDM.std()

    fMRI = Representation(
        name=f"fMRI_{roi}",
        sim_mat=mean_RDM,
        get_embedding=False,
        object_labels=object_labels,
        category_name_list=new_category_name_list,
        num_category_list=category_num_list,
        category_idx_list=category_idx_list,
        func_for_sort_sim_mat=sort_matrix_with_categories
    )

    # models
    # load model data
    model_list = [
        'AlexNet', 
        'VGG19',
        'CLIP_B16_OpenAI',
        #'CLIP_B16_datacomp_l_s1b_b8k',
        #'CLIP_B16_datacomp_xl_s13b_b90k',
        #'CLIP_B16_laion2B-s34B-b88K', 
        # 'CLIP_L14_commonpool_xl_laion_s13b_b90k', 
        'ViT_B16_ImageNet1K', 
        #'ViT_B16_ImageNet21K',
    ]

    layer_numbers = {'AlexNet': 5, 'VGG19': 10, 'ViT_B16_ImageNet1K': 13, 'ViT_B16_ImageNet21K': 13, 'CLIP_B16_OpenAI': 12}
    
    #representations =  [fMRI]
    name_list = ['fMRI'] + model_list

    for model_name in model_list:
        layer_number = layer_numbers[model_name]
        for layer in range(layer_number):
            if layer_wise:
                data_path = f"../data/models/shared515/sim_mat/sorted_{model_name.lower()}_{layer}_conv.pt"
            else:
                data_path = f"../data/models/shared515/sim_mat/sorted_{model_name.lower()}_conv.pt"
                layer = 'final'
            sim_mat = torch.load(data_path).to('cpu').numpy()

            # Z-transform
            if z_transform:
                sim_mat = (sim_mat - sim_mat.mean()) / sim_mat.std()

            model = Representation(
                name = f"{model_name}_{layer}", 
                sim_mat = sim_mat,
                get_embedding=False,
                object_labels=object_labels,
                category_name_list=new_category_name_list,
                num_category_list=category_num_list,
                category_idx_list=category_idx_list,
                func_for_sort_sim_mat=sort_matrix_with_categories
            )

            representations = [fMRI, model]

            main_results_dir = f"../results/gw_alignment/models/{'ztransform/' if z_transform else ''}"
            init_mat_plan = 'random'
            data_name = f"NSD_{roi}_vs_{model_name}_{layer}{'_ztransform' if z_transform else ''}"

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
                metric="cosine",
                main_results_dir=main_results_dir,
                data_name=data_name, 
                histogram_matching=True
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

            fig_dir = f"../results/figs/{roi}/models/{model_name}/layer{layer}/{'ztransform/' if z_transform else ''}"
            os.makedirs(fig_dir, exist_ok=True)

            alignment.show_sim_mat(
                sim_mat_format="sorted",
                visualization_config=vis_config, 
                show_distribution=True,
                fig_dir=fig_dir
                )
            alignment.RSA_get_corr()
            rsa_corr = pd.DataFrame([alignment.RSA_corr], index=['correlation'])
            df_rsa = pd.concat([df_rsa, rsa_corr], axis=1)
            

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

            OT_sorted = alignment.gw_alignment(
                compute_OT=compute_OT,
                delete_results=False,
                OT_format="sorted",
                return_data=True,
                save_dataframe=True,
                visualization_config=vis_config_OT,
                fig_dir=fig_dir
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
            
            if not layer_wise:
                break
    
    top_k_accuracy = top_k_accuracy.T
    top_k_accuracy.index.name = 'pair_name'
    top_k_accuracy_all = pd.concat([top_k_accuracy_all, top_k_accuracy], axis=0)
    
    cat_accuracy = cat_accuracy.T
    cat_accuracy.index.name = 'pair_name'
    cat_accuracy_all = pd.concat([cat_accuracy_all, cat_accuracy], axis=0)
    
    df_rsa = df_rsa.T
    df_rsa.index.name = 'pair_name'
    df_rsa_all = pd.concat([df_rsa_all, df_rsa], axis=0)
    
    df_gwd = df_gwd.T
    df_gwd.index.name = 'pair_name'
    df_gwd_all = pd.concat([df_gwd_all, df_gwd], axis=0)

#%%
### save
top_k_accuracy_all.to_csv(os.path.join(main_results_dir, 'top_k_accuracy_all_pairs.csv'))
cat_accuracy_all.to_csv(os.path.join(main_results_dir, 'cat_accuracy_all_pairs.csv'))
df_rsa_all.to_csv(os.path.join(main_results_dir, 'df_rsa_all_pairs.csv'))
df_gwd_all.to_csv(os.path.join(main_results_dir, 'df_gwd_all_pairs.csv'))

### show summary figure
    
#%%
main_results_dir = "../results/gw_alignment/models/"
fig_dir = f"../results/figs/models/"

# top k acc
df = pd.read_csv(os.path.join(main_results_dir, 'top_k_accuracy_all_pairs.csv'))
top_k = 1
title = f'top{top_k} matching rate'
file_name = f'top_{top_k}_accuracy'

show_matrix(df, col_name=str(top_k), title=title, save_dir=fig_dir, file_name=file_name)
# %%
# category
df = pd.read_csv(os.path.join(main_results_dir, 'cat_accuracy_all_pairs.csv')) 
top_k = 1
title = f'category level accuracy'
file_name = f'category_top_{top_k}_accuracy'

show_matrix(df, col_name=str(top_k), title=title, save_dir=fig_dir, file_name=file_name)

#%%
# rsa
df = pd.read_csv(os.path.join(main_results_dir, 'df_rsa_all_pairs.csv'))
col_name = 'correlation'
title = 'RSA correlation'
file_name = f'rsa_all_pairs'

show_matrix(df, col_name=col_name, title=title, save_dir=fig_dir, file_name=file_name)

# %%
# gwd
df = pd.read_csv(os.path.join(main_results_dir, 'df_gwd_all_pairs.csv'))
col_name = 'gwd'
title = 'GWD'
file_name = f'gwd_all_pairs'

show_matrix(df, col_name=col_name, title=title, save_dir=fig_dir, file_name=file_name)
# %%
