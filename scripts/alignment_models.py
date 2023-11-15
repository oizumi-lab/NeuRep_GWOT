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
roi_list = ['pVTC', 'aVTC', 'v1', 'v2', 'v3'] #['pVTC', 'aVTC', 'v1', 'v2', 'v3']

compute_OT = True

# category data
category_mat = pd.read_csv("../data/category_mat_shared515.csv")
object_labels, category_idx_list, category_num_list, new_category_name_list = get_category_data(category_mat)

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
        'CLIP_B16_datacomp_l_s1b_b8k',
        'CLIP_B16_datacomp_xl_s13b_b90k',
        'CLIP_B16_laion2B-s34B-b88K', 
        # 'CLIP_L14_commonpool_xl_laion_s13b_b90k', 
        'ViT_B16_ImageNet1K', 
        'ViT_B16_ImageNet21K',
    ]

    #representations =  [fMRI]
    name_list = ['fMRI'] + model_list

    for model_name in model_list:
        data_path = "../data/models/shared515/sim_mat/sorted_" + model_name.lower() + "_conv.pt"
        sim_mat = torch.load(data_path).to('cpu').numpy()

        model = Representation(
            name = model_name, 
            sim_mat = sim_mat,
            get_embedding=False,
            object_labels=object_labels,
            category_name_list=new_category_name_list,
            num_category_list=category_num_list,
            category_idx_list=category_idx_list,
            func_for_sort_sim_mat=sort_matrix_with_categories
        )

        representations = [fMRI, model]
        
        opt_config = OptimizationConfig(
            init_mat_plan="random",
            db_params={"drivername": "sqlite"},
            num_trial=100,
            n_iter=1, 
            max_iter=1000,
            sampler_name="tpe", 
            eps_list=[1e-5, 1e-1],
            eps_log=True,
            to_types='torch',
            device='cuda',
            multi_gpu=['cuda:0', 'cuda:1']
            )

        alignment = AlignRepresentations(
            config=opt_config, 
            representations_list=representations,
            metric="cosine",
            main_results_dir="../results/gw_alignment/models/",
            data_name=f"NSD_{roi}_vs_{model_name}", 
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

        os.makedirs(f"../results/figs/{roi}/models/{model_name}", exist_ok=True)

        alignment.show_sim_mat(
            sim_mat_format="sorted",
            visualization_config=vis_config, 
            show_distribution=False,
            fig_dir=f"../results/figs/{roi}/models/{model_name}"
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

        OT_sorted = alignment.gw_alignment(
            compute_OT=compute_OT,
            delete_results=False,
            OT_format="sorted",
            return_data=True,
            visualization_config=vis_config_OT,
            fig_dir=f"../results/figs/{roi}/models/{model_name}"
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
            fig_dir=f"../results/figs/{roi}/models/{model_name}"
            )

        ## Calculate the accuracy of the optimized OT matrix
        top_k_list = [1, 3, 5]

        alignment.calc_accuracy(
            top_k_list=top_k_list, 
            eval_type="ot_plan"
            )

        alignment.plot_accuracy(
            eval_type="ot_plan", 
            fig_dir=f"../results/figs/{roi}/models/{model_name}",
            fig_name="accuracy_ot_plan.png"
            )

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
            fig_dir=f"../results/figs/{roi}/models/{model_name}",
            fig_name="category_level_accuracy_ot_plan.png"
            )
# %%
