#%%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../'))

import numpy as np
import pandas as pd
import pickle as pkl
import torch

from GW_methods.src.align_representations import Representation, AlignRepresentations, OptimizationConfig, VisualizationConfig

#%%
compute_OT = True

subj_list = [f"subj0{i+1}" for i in range(8)]
roi_list = ["pVTC"]

for roi in roi_list:
    representations = []
    for subj in subj_list:
        embedding = np.load(f"/home1/data/common-data/natural-scenes-dataset/rsa/roi_analyses/{subj}_{roi}_fullrdm_correlation.npy")
        representation = Representation(
            name=f"{subj}_{roi}",
            embedding=embedding,
            metric="euclidean"
        )
        representations.append(representation)

    opt_config = OptimizationConfig(
        init_mat_plan="random",
        db_params={"drivername": "sqlite"},
        num_trial=100,
        n_iter=1, 
        max_iter=200,
        sampler_name="tpe", 
        eps_list=[0.02, 0.2],
        eps_log=True,
        )
            
    alignment = AlignRepresentations(
        config=opt_config, 
        representations_list=representations,
        metric="euclidean",
        main_results_dir="../results/gw_alignment/",
        data_name=f"NSD_within_{roi}", 
        )
    
    vis_config = VisualizationConfig(
        figsize=(8, 6), 
        #title_size = 15, 
        cmap = "rocket",
        #cbar_ticks_size=30,
        font = "Arial",
        cbar_label="Dissimilarity",
        #cbar_label_size=40,
        )
    
    alignment.show_sim_mat(
        visualization_config=vis_config, 
        show_distribution=False,
        fig_dir=f"../results/figs/{roi}/"
        )
    alignment.RSA_get_corr()
    
    vis_config_OT = VisualizationConfig(
        figsize=(8, 6), 
        title_size = 15, 
        cmap = "rocket",
        #cbar_ticks_size=30,
        font = "Arial",
        cbar_label="Probability",
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
        visualization_config=vis_config_OT,
        fig_dir=f"../results/figs/{roi}/"
        )
    
# %%
