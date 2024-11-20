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
import itertools

from src.utils import sample_participants, split_lists
from GW_methods.src.align_representations import Representation, AlignRepresentations, OptimizationConfig, VisualizationConfig
from GW_methods.src.utils.utils_functions import get_category_data, sort_matrix_with_categories

#%%
n_subj = 8
n_groups = 2
subj_list = [f"subj0{i+1}" for i in range(8)]
#roi_list = ['hV4'] #['pVTC', 'aVTC', 'v1', 'v2', 'v3']
# roi_list = ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]
roi_list = ['pVTC', 'aVTC', 'v1', 'v2'] # cuda:0
# roi_list = ['v3', 'OPA', 'PPA', 'RSC', 'MTL'] # cuda:2
#roi_list = ['thalamus', 'MTL']
n_sample = 10
seed_list = range(n_sample)
#seed_list = range(5, 10)
#seed_list = [4]

one_vs_one = False
if one_vs_one:
    seed_list = [0]

RDM_concat = True

compute_OT = True
# device = 'cuda:2'
device = 'cuda:0'
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
category_mat = pd.read_csv("../data/category_mat_shared515.csv", index_col=0)
object_labels, category_idx_list, category_num_list, new_category_name_list = get_category_data(category_mat)
    
#%%
for roi in roi_list:
    ### set dataframes
    top_k_list = [1, 3, 5]
    top_k_accuracy = pd.DataFrame()
    #top_k_accuracy["top_n"] = top_k_list
    
    k_nearest_accuracy = pd.DataFrame()
    sup_accuracy = pd.DataFrame()
    
    cat_accuracy = pd.DataFrame()
    #cat_accuracy["top_n"] = top_k_list
    
    df_rsa = pd.DataFrame()
    df_gwd = pd.DataFrame()
    
    for seed_id, groups in enumerate(groups_list):
        seed = seed_list[seed_id]
        
        representations = []
        for j, group in enumerate(groups):
            
            if RDM_concat:
                mean_RDM = np.load(f"/mnt/NAS/common_data/natural-scenes-dataset/rsa/roi_analyses/seed{seed}_group{j}_{roi}_fullrdm_shared515_correlation.npy")
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

        
        #main_results_dir = "../results/gw_alignment/"
        main_results_dir = "/mnt/NAS/user_data/ken-takeda/GWOT/Takeda_NSD/gw_alignment"
        init_mat_plan = 'random'
        
        concat_or_not = '_concat' if RDM_concat else ''
        
        if one_vs_one:
            data_name = f"NSD_within_{roi}_one_vs_one"
            pair_name = f"subj{groups[0][0]}_vs_subj{groups[1][0]}"
        else:
            data_name = f"NSD_within_{roi}_seed{seed}{concat_or_not}"
            pair_name = f"Group1_{roi}_vs_Group2_{roi}"
        
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

        if one_vs_one:
            fig_dir = f"../results/figs/{roi}/one_vs_one/"
        else:
            fig_dir = f"../results/figs/{roi}/seed{seed}{concat_or_not}/"
        
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

        #%%
        vis_config_OT = VisualizationConfig(
            figsize=(8, 6), 
            #title_size = 15, 
            cmap = "rocket_r",
            cbar_ticks_size=30,
            font = "Arial",
            cbar_label="Probability",
            cbar_label_size=40,
            #color_labels = new_color_order,
            #color_label_width = 5,
            xlabel=f"515 images of Group1",
            xlabel_size = 35,
            ylabel=f"515 images of Group2",
            ylabel_size = 35,
            colorbar_range=[0, 0.001]
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
        
        OT = alignment.gw_alignment(
            compute_OT=compute_OT,
            delete_results=False,
            OT_format="default",
            return_data=True,
            return_figure=False,
            )
        
        np.save(os.path.join(main_results_dir, data_name+'_'+pair_name, init_mat_plan, 'data/gw_best'), OT)
        
        # record gwd
        gwds = {}
        for pairwise in alignment.pairwise_list:
            pair_name = pairwise.pair_name
            
            study_name = data_name + '_' + pair_name
            df = pd.read_csv(os.path.join(main_results_dir, study_name, init_mat_plan, study_name+'.csv'))
            gwds[pair_name] = df['value'].min()
        gwds = pd.DataFrame([gwds], index=['gwd'])
        df_gwd = pd.concat([df_gwd, gwds], axis=1)

        #%%
        vis_config_log = VisualizationConfig(
            figsize=(8, 6), 
            title_size = 10, 
            cbar_ticks_size=30,
            font = "Arial",
            fig_ext="svg",
            xlabel_size=35,
            xticks_size=30,
            xticks_rotation=0,
            ylabel_size=35,
            yticks_size=30,
            cbar_label_size=30,
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
        
        if get_embedding:
            alignment.calc_accuracy(
                top_k_list=top_k_list, 
                eval_type="k_nearest",
                )

            k_nearest_accuracy = pd.concat([k_nearest_accuracy, alignment.k_nearest_matching_rate])

            alignment.calc_accuracy(
                top_k_list=top_k_list,
                eval_type="k_nearest",
                ot_to_evaluate=np.eye(515)
            )

            sup_accuracy = pd.concat([sup_accuracy, alignment.k_nearest_matching_rate])
        

        alignment.plot_accuracy(
            eval_type="ot_plan", 
            fig_dir=fig_dir, 
            fig_name="accuracy_ot_plan.png"
            )
        
        top_k_accuracy = pd.concat([top_k_accuracy, alignment.top_k_accuracy])
        if get_embedding:
            k_nearest_accuracy = pd.concat([k_nearest_accuracy, alignment.k_nearest_matching_rate])
            sup_accuracy = pd.concat([sup_accuracy, alignment.k_nearest_matching_rate])
        
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
        
        cat_accuracy = pd.concat([cat_accuracy, alignment.top_k_accuracy])
        
        #%%
        if get_embedding:
            vis_config_emb = VisualizationConfig(
                figsize=(8, 8),
                xlabel="PC1",
                ylabel="PC2",
                zlabel="PC3",
                marker_size=15,
                legend_size=10,
                )

            category_name_list = ['bird', 'giraffe', 'chair', 'clock', 'bottle', 'elephant']
            object_labels, category_idx_list, num_category_list, category_name_list = get_category_data(category_mat, category_name_list, show_numbers = True)

            alignment.visualize_embedding(
                dim=3,
                visualization_config=vis_config_emb,
                fig_dir=fig_dir,
                category_name_list=category_name_list,
                category_idx_list=category_idx_list,
                num_category_list=num_category_list,

            )
        
        #%%
    # save data
    save_dir = f'../results/gw_alignment/within{roi}/'
    os.makedirs(save_dir, exist_ok=True)

    top_k_accuracy.to_csv(os.path.join(save_dir, 'top_k_accuracy.csv'))
    cat_accuracy.to_csv(os.path.join(save_dir, 'category_accuracy.csv'))
    if get_embedding:
        k_nearest_accuracy.to_csv(os.path.join(save_dir, 'k_nearest_accuracy.csv'))
        sup_accuracy.to_csv(os.path.join(save_dir, 'sup_accuracy.csv'))
    
    df_rsa = df_rsa.T
    df_rsa.index.name = 'pair_name'
    df_rsa.to_csv(os.path.join(save_dir, 'rsa_correlation.csv'))
    
    df_gwd = df_gwd.T
    df_gwd.index.name = 'pair_name'
    df_gwd.to_csv(os.path.join(save_dir, 'gw_distance.csv'))
    
# %%


### make summary figures

# top k acc
top_k_list = [1, 3, 5]
for roi in roi_list:
    save_dir = f'../results/gw_alignment/within{roi}/'
    df_acc = pd.read_csv(os.path.join(save_dir, 'top_k_accuracy.csv'))
    
    # for each k
    #for k in top_k_list:
    #    df_filtered = df_acc[df_acc['top_n'] == k]
    pair_name = f'Group1_{roi}_vs_Group2_{roi}'
    
    # plot
    labels = [str(n) for n in top_k_list]

    palette = sns.color_palette("bright", n_colors=len(labels))

    plt.figure(figsize=(6, 6))
    sns.swarmplot(x='top_n', y=pair_name, data=df_acc, palette=palette, size=8)
    
    plt.xlabel('Top-k', size=35)
    plt.ylabel('Matching Rate', size=35)
    plt.xticks(ticks=[0, 1, 2], labels=labels)
    plt.xticks(size=30)
    plt.yticks(size=30)
    # ylim [0,100]
    plt.ylim([-5, 105])
    plt.tight_layout()
    plt.title(f'{roi}', size=35)
    plt.savefig(f"../results/figs/{roi}/top_k_accuracy_swarmplot.png")
    plt.show()

#%%
# top1
top_k = 1
all_data = pd.DataFrame()
for roi in roi_list:
    save_dir = f'../results/gw_alignment/within{roi}/'
    df_acc = pd.read_csv(os.path.join(save_dir, 'top_k_accuracy.csv'))
    
    pair_name = f'Group1_{roi}_vs_Group2_{roi}'
    
    df_acc = df_acc[df_acc['top_n'] == top_k]
    data = [[pair_name, acc] for acc in df_acc[pair_name]]
    df = pd.DataFrame(data=data, columns=['pair_name', 'accuracy'])
    
    all_data = pd.concat([all_data, df], axis=0)

# plot
labels = roi_list

palette = sns.color_palette("bright", n_colors=len(labels))

plt.figure(figsize=(6, 6))
sns.swarmplot(x='pair_name', y='accuracy', data=all_data, palette=palette, size=8)

plt.xlabel('roi', size=35)
plt.ylabel(f'top {top_k} matching rate', size=35)
plt.xticks(ticks=range(len(labels)), labels=labels)
plt.xticks(size=15)
plt.yticks(size=30)
# ylim [0,100]
plt.ylim([-5, 105])
plt.tight_layout()
plt.savefig(f"../results/figs/within_roi/top_{top_k}_accuracy_swarmplot.png")
plt.show()

#%%
# top1
top_k = 1
all_data = pd.DataFrame()
for roi in roi_list:
    save_dir = f'../results/gw_alignment/within{roi}/'
    df_acc = pd.read_csv(os.path.join(save_dir, 'category_accuracy.csv'))
    
    pair_name = f'Group1_{roi}_vs_Group2_{roi}'
    
    df_acc = df_acc[df_acc['top_n'] == top_k]
    data = [[pair_name, acc] for acc in df_acc[pair_name]]
    df = pd.DataFrame(data=data, columns=['pair_name', 'accuracy'])
    
    all_data = pd.concat([all_data, df], axis=0)

# plot
labels = roi_list

palette = sns.color_palette("bright", n_colors=len(labels))

plt.figure(figsize=(6, 6))
sns.swarmplot(x='pair_name', y='accuracy', data=all_data, palette=palette, size=8)

plt.xlabel('roi', size=35)
plt.ylabel(f'top {top_k} matching rate', size=35)
plt.xticks(ticks=range(len(labels)), labels=labels)
plt.xticks(size=15)
plt.yticks(size=30)
# ylim [0,100]
plt.ylim([-5, 105])
plt.tight_layout()
plt.savefig(f"../results/figs/within_roi/category_top_{top_k}_accuracy_swarmplot.png")
plt.show()

#%%
# rsa
all_data = pd.DataFrame()
for roi in roi_list:
    save_dir = f'../results/gw_alignment/within{roi}/'
    df_acc = pd.read_csv(os.path.join(save_dir, 'rsa_correlation.csv'), index_col=0)
    
    pair_name = f'Group1_{roi}_vs_Group2_{roi}'
    all_data = pd.concat([all_data, df_acc], axis=0)


# plot
labels = roi_list

palette = sns.color_palette("bright", n_colors=len(labels))

plt.figure(figsize=(6, 6))
sns.swarmplot(x='pair_name', y='correlation', data=all_data, palette=palette, size=8)

plt.xlabel('roi', size=35)
plt.ylabel('RSA correlation', size=35)
plt.xticks(ticks=range(len(labels)), labels=labels)
plt.xticks(size=15)
plt.yticks(size=30)
# ylim [0,100]
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig(f"../results/figs/within_roi/rsa_swarmplot.png")
plt.show()
# %%

# gwd
all_data = pd.DataFrame()
for roi in roi_list:
    save_dir = f'../results/gw_alignment/within{roi}/'
    df_acc = pd.read_csv(os.path.join(save_dir, 'gw_distance.csv'), index_col=0)
    
    pair_name = f'Group1_{roi}_vs_Group2_{roi}'
    all_data = pd.concat([all_data, df_acc], axis=0)


# plot
labels = roi_list

palette = sns.color_palette("bright", n_colors=len(labels))

plt.figure(figsize=(6, 6))
sns.swarmplot(x='pair_name', y='gwd', data=all_data, palette=palette, size=8)

plt.xlabel('roi', size=35)
plt.ylabel('GWD', size=35)
plt.xticks(ticks=range(len(labels)), labels=labels)
plt.xticks(size=15)
plt.yticks(size=15)
# ylim [0,100]
plt.ylim([0, 0.02])
plt.tight_layout()
plt.savefig(f"../results/figs/within_roi/gwd_swarmplot.png")
plt.show()
# %%
