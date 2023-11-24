#%%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../'))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
# load results
data_dir = '../results/gw_alignment/'
roi_list = ['pVTC', 'aVTC', 'v1', 'v2', 'v3']

all_data = pd.DataFrame(columns=['roi1', 'roi2', 'rsa_corr', 'top1_acc', 'category_top1', 'gwd'])

# within roi
for roi in roi_list:
    df_rsa = pd.read_csv(data_dir+f'within{roi}/rsa_correlation.csv')
    rsa_corr = df_rsa['correlation'].mean()
    
    df_gwd = pd.read_csv(data_dir+f'within{roi}/gw_distance.csv')
    gwd = df_gwd['gwd'].mean()
    
    df_acc = pd.read_csv(data_dir+f'within{roi}/top_k_accuracy.csv')
    top1_acc = df_acc[df_acc['top_n'] == 1].iloc[:, 1].mean()
    
    df_category = pd.read_csv(data_dir+f'within{roi}/category_accuracy.csv')
    category_top1 = df_category[df_category['top_n'] == 1].iloc[:, 1].mean()
    
    data = {
        'roi1': roi,
        'roi2': roi,
        'rsa_corr': rsa_corr,
        'top1_acc': top1_acc,
        'category_top1': category_top1,
        'gwd': gwd
        }
    
    all_data = all_data.append(data, ignore_index=True)
#%%

# across roi
df_rsa = pd.read_csv(data_dir+f'across_roi/rsa_correlation.csv')
av_rsa_corr = df_rsa.groupby('pair_name')['correlation'].mean().rename('rsa_corr')

df_gwd = pd.read_csv(data_dir+f'across_roi/gw_distance.csv')
av_gwd = df_gwd.groupby('pair_name')['gwd'].mean()

df_acc = pd.read_csv(data_dir+f'across_roi/top_k_accuracy.csv', index_col=1)
av_top1_acc = df_acc['1'].groupby('pair_name').mean().rename('top1_acc')

df_category = pd.read_csv(data_dir+f'across_roi/category_accuracy.csv', index_col=1)
av_category_top1 = df_category['1'].groupby('pair_name').mean().rename('category_top1')

data = pd.concat([av_rsa_corr, av_top1_acc, av_category_top1, av_gwd], axis=1).reset_index()
data[['roi1', 'roi2']] = data['pair_name'].str.split('_vs_', expand=True)

all_data = all_data.append(data, ignore_index=True)
del all_data['pair_name']
all_data.to_csv(data_dir+'result_across_roi.csv')
# %%
# visualize the result as a matrix
all_data = pd.read_csv(data_dir+'result_across_roi.csv', index_col=0)
result_list = ['rsa_corr', 'top1_acc', 'category_top1', 'gwd']
fig_dir = '../results/figs/across_roi/'

for result in result_list:
    # get the matrix from the dataframe and make it symmetric
    matrix = all_data.pivot(index='roi1', columns='roi2', values=result).fillna(0)
    
    # make the matrix symmetric
    matrix = matrix + matrix.T - np.diag(np.diag(matrix))
    
    plt.figure()
    sns.heatmap(matrix, annot=True, cmap='rocket')
    plt.title(result.replace('_', ' '))
    plt.show()
    plt.savefig(fig_dir+f'{result}.png')


# %%
