# %%
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pickle
from scipy.spatial import distance
# nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
# screenコマンドのログを保存する：　Ctrlキ+Aキー → Shift+Hキー
# sudo -u postgres psql
import re
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text) ]


# %%
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
#from src.utils import make_graph_for_eval

# %%
os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
#from GW_methods.src import align_representations, compute_alignment
from GW_methods.src.utils.utils_functions import get_category_data, sort_matrix_with_categories


# %%
category_mat = pd.read_csv("../data/category_mat_shared515.csv", index_col=0)
object_labels, category_idx_list, num_category_list, category_name_list = get_category_data(category_mat)


#%%
# ot_path は手動で指定
roi_list = ['v1', 'v2', 'v3', 'pVTC', 'aVTC', 'OPA', 'PPA', 'RSC', 'MTL']

concat = True
concat_or_not = '_concat' if concat else ''

for roi in roi_list:
    #if roi == 'v1':
    #    vmax = 0.0001
    #elif roi == 'v2':
    #    vmax = 0.0001
    #elif roi == 'v3':
    #    vmax = 0.0001
    #elif roi == 'pVTC':
    #    vmax = 0.00001
    #elif roi == 'aVTC':
    #    vmax = 0.00001
    # vmax = 0.00001
    vmax = 0.000001
        
    seed=7

    main_results_dir = "/mnt/NAS/user_data/ken-takeda/GWOT/Takeda_NSD/gw_alignment"
    init_mat_plan = 'random'
    data_name = f"NSD_within_{roi}_seed{seed}{concat_or_not}"
    pair_name = f"Group1_{roi}_vs_Group2_{roi}"

    fig_dir = f"../results/figs/{roi}/seed{seed}{concat_or_not}/"

    ot_data = os.path.join(main_results_dir, data_name+'_'+pair_name, init_mat_plan, 'data/gw_best.npy')
    #target = "Supervised"
    #ot_path = f"/home1/data/masaru-sasaki/THINGS/Takahashi_Test/Takahashi_Test_behavior_66d_vs_SimCLRv2_RN101_2x_{target}/random/data/"
    #
    #ot_data = sorted(os.listdir(ot_path), key=natural_keys)[-1]
    #ot_data = os.path.join(ot_path, ot_data)

    
    #ot = torch.load(ot_data).detach().cpu().numpy()
    ot = np.load(ot_data)[0]

    sorted_ot = sort_matrix_with_categories(ot, category_idx_list)

    
    # enlarged_list = ['dining table', 'chair', 'bird', 'elephant']
    # enlarged_list = ['bird', 'elephant', 'giraffe', 'car', 'airplane', 'motorcycle']
    # enlarged_list = ['bird', 'elephant', 'giraffe', 'car', 'airplane', 'motorcycle', 'chair', 'dining table']
    enlarged_list = ['elephant', 'giraffe', 'chair', 'dining table']
    enlarged_idx_list = []
    category_boundary_idx = []
    for category_name in enlarged_list:
        idx = category_name_list.index(category_name)
        enlarged_idx_list.extend(category_idx_list[idx])
        category_boundary_idx.append(len(enlarged_idx_list))
    enlarged_ot = ot[np.ix_(enlarged_idx_list, enlarged_idx_list)]
    enlarged_object_labels = np.array(category_mat.index)[enlarged_idx_list]

    extent = [0, len(enlarged_object_labels), 0, len(enlarged_object_labels)]

    xpos = [0] + category_boundary_idx
    x1 = np.roll(xpos, -1)  # = np.ndarray([ x[1], x[2], ..., x[n-1], x[0] ])
    ave = (xpos + x1)/2.    # これは長さ n の ndarray で最後の要素 (x[0]+x[n-1])/2. は要らない
    text_xpos = ave[:-1]
    text_xpos[-1] = text_xpos[-1] + 3.5

    plt.style.use("default")
    plt.rcParams["grid.color"] = "black"
    plt.rcParams['font.family'] = "Sans-serif"

    plt.figure(figsize=(8, 6))
    plt.imshow(
        enlarged_ot,
        cmap='rocket_r',
        vmin=0,
        vmax=vmax,
        extent=extent,
        )

    for i in range(len(category_boundary_idx)):
        plt.axvline(x=category_boundary_idx[i], color='blue', linewidth=1, linestyle='--')
        plt.axhline(y=(len(enlarged_idx_list)-category_boundary_idx[i]), color='black', linewidth=1, linestyle='--')

        plt.text(
            text_xpos[i] - len(enlarged_list[i]) - 1, 
            len(enlarged_idx_list) - xpos[i] + 1, 
            enlarged_list[i], 
            fontsize=12,
            weight='bold',
            color='blue',
        )

    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Probability')
    # plt.show()
    plt.savefig(f"{fig_dir}/enlarged_OT.png", dpi=300, bbox_inches='tight')
# %%
