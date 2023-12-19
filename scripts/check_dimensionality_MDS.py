#%%
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

#%%
roi_list = ['v1', 'v2', 'v3', 'pVTC', 'aVTC']
n_subj = 8
subj_list = [f"subj0{i+1}" for i in range(8)]
dim_list = range(10, 500, 50)

#%%
# load RDM and calculate mean RDM
RDMs = []
for roi in roi_list:
    RDM_roi = []
    for subj in subj_list:
        RDM = np.load(f"/home1/data/common-data/natural-scenes-dataset/rsa/roi_analyses/{subj}_{roi}_fullrdm_shared515_correlation.npy")
        RDM_roi.append(RDM)
    RDM_roi = np.mean(RDM_roi, axis=0)
    RDMs.append(RDM_roi)
    
#%%
# calculate MDS for each ROI and check the dimensionalities
# for given dimensions, calculate MDS and check the accuracy
# plot the tendency of accuracy for each ROI

#stress_data = {}
#stress_decay_data = {}
#for roi, RDM in zip(roi_list, RDMs):
#    stress_list = []
#    stress_decay_list = []
#    for dim in tqdm(dim_list):
#        mds = MDS(n_components=dim, metric=True, n_jobs=-1, dissimilarity='precomputed', normalized_stress='auto')
#        mds.fit(RDM)
#        stress_list.append(mds.stress_)
#        
#        # calculate stress decay
#        if len(stress_list) > 1:
#            stress_decay = (stress_list[-2] - stress_list[-1]) / stress_list[-2]
#            stress_decay_list.append(stress_decay)
#            
#    stress_data[roi] = stress_list
#    stress_decay_data[roi] = stress_decay_list
#
## save the stress data
#with open('../results/stress_data.pkl', 'wb') as f:
#    pickle.dump(stress_data, f)
#
#with open('../results/stress_decay_data.pkl', 'wb') as f:
#    pickle.dump(stress_decay_data, f)
## %%
## load the stress data
#with open('../results/stress_data.pkl', 'rb') as f:
#    stress_data = pickle.load(f)
#with open('../results/stress_decay_data.pkl', 'rb') as f:
#    stress_decay_data = pickle.load(f)
#
##%%
## plot the tendency of stress
#for roi in roi_list:
#    stress_list = stress_data[roi]
#    plt.figure()
#    plt.style.use('seaborn-v0_8-darkgrid')
#    plt.plot(dim_list, stress_list)
#    plt.xlabel('dimension')
#    plt.ylabel('stress')
#    plt.title(f'{roi}')
#    plt.savefig(f'../results/figs/{roi}/stress_{roi}.png')
#    plt.show()
#    
#    # plot the tendency of stress decay
#    stress_decay_list = stress_decay_data[roi]
#    plt.figure()
#    plt.style.use('seaborn-v0_8-darkgrid')
#    plt.plot(dim_list[1:], stress_decay_list)
#    # plot the horizontal line at 0.0
#    # plot the vertical line at the dimension where the stress decay is lower than 0.01
#    plt.axhline(0.05, color='r', linestyle='--')
#    for i, stress_decay in enumerate(stress_decay_list):
#        if stress_decay < 0.05:
#            plt.axvline(dim_list[i+1], color='r', linestyle='--')
#            break
#    
#    plt.xlabel('dimension')
#    plt.ylabel('stress decay')
#    plt.title(f'{roi}')
#    plt.savefig(f'../results/figs/{roi}/stress_decay_{roi}.png')
#    plt.show()
# %%
# check the distribution od MDS
for roi, RDM in zip(roi_list, RDMs):
    # extract the lowere triangular part of RDM
    RDM = np.tril(RDM, k=-1)
    RDM = RDM.flatten()
    
    plt.figure()
    plt.hist(RDM, bins=100)
    plt.title(f'{roi}')
    plt.savefig(f'../results/figs/{roi}/distribution_{roi}.png')
    plt.show()
    
# %%
# calculate dendrogram for each ROI
for roi, RDM in zip(roi_list, RDMs):
    # calculate linkage
    # replace the diagonal elements with 0.0
    np.fill_diagonal(RDM, 0.0)
    Z = linkage(squareform(RDM), method='ward')
    
    # plot dendrogram
    plt.figure(figsize=(25, 10))
    plt.title(f'{roi}')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    plt.xticks([])
    plt.yticks(size=20)
    dendrogram(Z)
    plt.savefig(f'../results/figs/{roi}/dendrogram_{roi}.png')
    plt.show()

# %%
