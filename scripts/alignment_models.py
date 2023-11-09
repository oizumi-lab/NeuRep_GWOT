#%%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../'))

import numpy as np
import pandas as pd
import pickle as pkl
import torch
import random

from GW_methods.src.align_representations import Representation, AlignRepresentations, OptimizationConfig, VisualizationConfig

#%%
n_subj = 8
roi_list = ["pVTC"] #['pVTC', 'aVTC', 'v1', 'v2', 'v3']

compute_OT = True

#%%
# fMRI
roi = roi_list[0]
RDMs = []
for i in range(n_subj):
    RDM = np.load(f"/home1/data/common-data/natural-scenes-dataset/rsa/roi_analyses/subj0{i+1}_{roi}_fullrdm_shared515_correlation.npy")
    RDMs.append(RDM)
RDMs = np.stack(RDMs)
mean_RDM = np.mean(RDMs, axis=0)

fMRI = Representation(
    name=f"fMRI_{roi}",
    sim_mat=mean_RDM,
    get_embedding=False
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

representations =  [fMRI]
name_list = ['fMRI'] + model_list

for model_name in model_list[:]:
    data_path = "../data/shared_515/sim_mat/" + model_name.lower() + "_conv.pt"
    sim_mat = torch.load(data_path).to('cpu').numpy()
    
    representation = Representation(
        name = model_name, 
        sim_mat = sim_mat,
        get_embedding=False
    )
    representations.append(representation)

opt_config = OptimizationConfig(
    init_mat_plan="random",
    db_params={"drivername": "sqlite"},
    num_trial=100,
    n_iter=1, 
    max_iter=200,
    sampler_name="tpe", 
    eps_list=[1e-5, 1e-3],
    eps_log=True,
    )

alignment = AlignRepresentations(
    config=opt_config, 
    representations_list=representations,
    metric="cosine",
    main_results_dir="../results/gw_alignment/",
    data_name=f"NSD_{roi}_vs_models", 
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

os.makedirs(f"../results/figs/{roi}/models/", exist_ok=True)

alignment.show_sim_mat(
    visualization_config=vis_config, 
    show_distribution=False,
    fig_dir=f"../results/figs/{roi}/models/"
    )
alignment.RSA_get_corr()
# %%
