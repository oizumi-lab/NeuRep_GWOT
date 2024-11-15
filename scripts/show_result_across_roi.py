#%%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../'))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

#%%
# load results
data_dir = '../results/gw_alignment/'
roi_list = ['v1', 'v2', 'v3', 'pVTC', 'aVTC']
#roi_list = ["early", "midventral", "ventral", "midlateral", "lateral", "midparietal", "parietal", "thalamus", "MTL"]

all_data = pd.DataFrame(columns=['roi1', 'roi2', 'rsa_corr', 'top1_acc', 'category_top1', 'gwd'])
df_for_plot = pd.DataFrame(columns=['roi', 'rsa_corr', 'top1_acc', 'category_top1', 'gwd'])

# within roi
for roi in roi_list:
    df_rsa = pd.read_csv(data_dir+f'within{roi}/rsa_correlation.csv')
    rsa_corr = df_rsa['correlation'].mean()
    rsa_list = df_rsa['correlation'].values.tolist()
    
    df_gwd = pd.read_csv(data_dir+f'within{roi}/gw_distance.csv')
    gwd = df_gwd['gwd'].mean()
    gwd_list = df_gwd['gwd'].values.tolist()
    
    df_acc = pd.read_csv(data_dir+f'within{roi}/top_k_accuracy.csv')
    top1_acc = df_acc[df_acc['top_n'] == 1].iloc[:, 1].mean()
    top1_acc_list = df_acc[df_acc['top_n'] == 1].iloc[:, 1].values.tolist()
    
    df_category = pd.read_csv(data_dir+f'within{roi}/category_accuracy.csv')
    category_top1 = df_category[df_category['top_n'] == 1].iloc[:, 1].mean()
    category_top1_list = df_category[df_category['top_n'] == 1].iloc[:, 1].values.tolist()
    
    data = {
        'roi1': roi,
        'roi2': roi,
        'rsa_corr': rsa_corr,
        'top1_acc': top1_acc,
        'category_top1': category_top1,
        'gwd': gwd
        }
    
    data_list = {
        'roi': [roi]*len(rsa_list),
        'rsa_corr': rsa_list,
        'top1_acc': top1_acc_list,
        'category_top1': category_top1_list,
        'gwd': gwd_list
        }
    
    all_data = all_data.append(data, ignore_index=True)
    df_for_plot = pd.concat([df_for_plot, pd.DataFrame(data_list)], ignore_index=True)
#%%
# within roi
# sworm plot
fig_dir = '../results/figs/within_roi/'
palette = sns.color_palette('bright', n_colors=5)
for result in ['rsa_corr', 'top1_acc', 'category_top1', 'gwd']:
    plt.figure(figsize=(8, 8))
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.swarmplot(x='roi', y=result, data=df_for_plot, palette=palette, size=8)
    #plt.title(result.replace('_', ' '))
    plt.xticks(ticks=range(len(roi_list)), labels=roi_list)
    plt.xlabel('roi', size=35)
    
    if result in ['rsa_corr']:
        ylabel = 'Correlation'
    elif result in ['top1_acc']:
        ylabel = 'Top-1 matching rate (%)'
    elif result in ['category_top1']:
        ylabel = 'Category matching rate (%)'
    elif result in ['gwd']:
        ylabel = 'GWD'
    plt.ylabel(ylabel, size=30)
    plt.xticks(size=20, rotation=45)
    plt.yticks(size=30)
    if result in ['rsa_corr']:
        plt.ylim(0, 1)
    elif result in ['top1_acc', 'category_top1']:
        plt.ylim(-5, 100)
        
        # show chance level
        if result == 'top1_acc':
            chance_level = 100/515
        elif result == 'category_top1':
            
            def calc_chance_level(category_mat):
                mat = category_mat.values
                n_object, n_category = mat.shape
                sum = np.sum(mat, axis=0)
                prob = sum / n_object
                chance = np.matmul(mat, prob)
                chance = np.sum(chance) / n_object

                return chance
            
            category_mat = pd.read_csv('../data/category_mat_shared515.csv', index_col=0)
            chance_level = calc_chance_level(category_mat) * 100
            print(chance_level)
        plt.axhline(y=chance_level, color='black', linestyle='--')
    plt.tight_layout()
    plt.show()
    plt.savefig(fig_dir+f'{result}_swarmplot.png')
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
    
    # sort the index and columns in the order of roi_list
    matrix = matrix.loc[roi_list, roi_list]
    
    # make the matrix symmetric
    matrix = matrix + matrix.T - np.diag(np.diag(matrix))
    
    plt.figure()
    # set font
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 20
    # do not show the xlabel title
    ax = sns.heatmap(matrix, annot=True, cmap='rocket_r', square=True)
    ax.set_xlabel('')
    ax.set_ylabel('')
    #sns.heatmap(matrix, annot=True, cmap='rocket', square=True)
    #plt.title(result.replace('_', ' '))
    plt.show()
    plt.savefig(fig_dir+f'{result}.png')

    # show a dendrogram
    plt.figure(figsize=(8, 4))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 30
    if result in ['top1_acc', 'category_top1', 'rsa_corr']:
        matrix = (1 - matrix)/100
    sch.dendrogram(sch.linkage(matrix, method='ward'), labels=roi_list)
    plt.xticks(size=30, rotation=45)
    plt.yticks(size=25)
    plt.ylabel('Distance', size=25)
    plt.tight_layout()
    plt.savefig(fig_dir+f'dendrogram_{result}.png')
    plt.show()


# %%
# show single group result
roi_list = ["early", "midventral", "ventral", "midlateral", "lateral", "midparietal", "parietal"]
df_rsa = pd.read_csv(data_dir+f'across_roi/single_group/rsa_correlation.csv')
av_rsa_corr = df_rsa.groupby('pair_name')['correlation'].mean().rename('rsa_corr')

df_gwd = pd.read_csv(data_dir+f'across_roi/single_group/gw_distance.csv')
av_gwd = df_gwd.groupby('pair_name')['gwd'].mean()

df_acc = pd.read_csv(data_dir+f'across_roi/single_group/top_k_accuracy.csv', index_col=1)
av_top1_acc = df_acc['1'].groupby('pair_name').mean().rename('top1_acc')

df_category = pd.read_csv(data_dir+f'across_roi/single_group/category_accuracy.csv', index_col=1)
av_category_top1 = df_category['1'].groupby('pair_name').mean().rename('category_top1')

data = pd.concat([av_rsa_corr, av_top1_acc, av_category_top1, av_gwd], axis=1).reset_index()
data[['roi1', 'roi2']] = data['pair_name'].str.split('_vs_', expand=True)

all_data = pd.DataFrame({
        'roi1': roi_list,
        'roi2': roi_list,
        'rsa_corr': [0]*len(roi_list),
        'top1_acc': [0]*len(roi_list),
        'category_top1': [0]*len(roi_list),
        'gwd': [0]*len(roi_list)
        })

all_data = all_data.append(data, ignore_index=True)
del all_data['pair_name']
all_data.to_csv(data_dir+'across_roi/single_group/result_across_roi.csv')
# %%
# visualize the result as a matrix
all_data = pd.read_csv(data_dir+'across_roi/single_group/result_across_roi.csv', index_col=0)
result_list = ['rsa_corr', 'top1_acc', 'category_top1', 'gwd']
fig_dir = '../results/figs/across_roi/single_group/'

for result in result_list:
    # get the matrix from the dataframe and make it symmetric
    matrix = all_data.pivot(index='roi1', columns='roi2', values=result).fillna(0)
    
    # sort the index and columns in the order of roi_list
    matrix = matrix.loc[roi_list, roi_list]
    
    # make the matrix symmetric
    matrix = matrix + matrix.T - np.diag(np.diag(matrix))
    
    plt.figure()
    # set font
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    # do not show the xlabel title
    ax = sns.heatmap(matrix, annot=True, cmap='rocket_r', square=True)
    ax.set_xlabel('')
    ax.set_ylabel('')
    #sns.heatmap(matrix, annot=True, cmap='rocket', square=True)
    #plt.title(result.replace('_', ' '))
    plt.show()
    plt.savefig(fig_dir+f'{result}.png')

    # show a dendrogram
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 20
    if result in ['top1_acc', 'category_top1', 'rsa_corr']:
        matrix = (1 - matrix)/100
    sch.dendrogram(sch.linkage(matrix, method='ward'), labels=roi_list)
    plt.xticks(size=20, rotation=45)
    plt.yticks(size=25)
    plt.ylabel('Distance', size=25)
    plt.tight_layout()
    plt.savefig(fig_dir+f'dendrogram_{result}.png')
    plt.show()
# %%
