import os
import random
import numpy as np
import pandas as pd
from scipy.spatial import distance
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def sample_participants(N, Z, seed):
    random.seed(seed)
    participants_list = random.sample(range(N), Z)
    return participants_list

def split_lists(list, n_groups): 
    N = len(list) // n_groups
    result = []
    for i in range(n_groups):
        list_i = list[N*i : N*(i+1)]
        result.append(list_i)
    return result

def get_meta_RDM(RDM_list, metric='correlation'):
    """
    function for getting meta RDM
    1. concatenate all RDMs -> get RDM_concat (shape of (n x m, n)
    2. compute meta RDM with RDM_concat

    Args:
        RDM_list (list): list of m RDMs. Each RDM is an np.ndarray of a shape (n, n).
    """
    RDM_concat = np.concatenate(RDM_list, axis=0)
    
    meta_RDM = distance.cdist(RDM_concat, RDM_concat, metric=metric)
    
    return meta_RDM

def sum_of_block_matrices(matrix, block_size):
    """
    Compute the sum of blocks of a matrix.

    Args:
        matrix (_type_): _description_
        block_size (int): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # Check if the matrix can be divided into blocks of size block_size
    if matrix.shape[0] % block_size != 0 or matrix.shape[1] % block_size != 0:
        raise ValueError("The matrix cannot be divided into blocks of size block_size.")
    
    # Compute the sum of blocks
    n_blocks = matrix.shape[0] // block_size # Number of blocks
    block_sum_matrix = np.zeros((n_blocks, n_blocks)) # Initialize the matrix for the sum of blocks
    
    for i in tqdm(range(n_blocks)):
        for j in range(n_blocks):
            block_sum_matrix[i, j] = np.sum(matrix[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size])

    return block_sum_matrix

def show_matrix(df, col_name, title, save_dir, file_name, first_items=None, second_items=None):
    # ペア名を分割して、別々の列にする
    df[['first_item', 'second_item']] = df['pair_name'].str.split('_vs_', expand=True)

    # マトリックスの初期化
    if first_items is None or second_items is None:
    #unique_items = np.unique(df[['first_item', 'second_item']].values)
        first_items = np.unique(df['first_item'].values)
        second_items = np.unique(df['second_item'].values)
    matrix = pd.DataFrame(np.nan, index=first_items, columns=second_items)

    # マトリックスにGWD値を代入
    for index, row in df.iterrows():
        matrix.at[row['first_item'], row['second_item']] = row[col_name]

    plt.figure(figsize=(8, 10))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 20
    sns.heatmap(matrix, annot=True, cmap='rocket', fmt=".2f")
    #plt.tick_params(axis='x', labelsize=15)
    #plt.tick_params(axis='y', labelsize=15)
    #plt.title(title, fontsize=20)
    plt.subplots_adjust(top=0.95, bottom=0.3)
    plt.show()
    plt.savefig(os.path.join(save_dir, file_name))
    
### check chance level
def calc_chance_level(category_mat):
    mat = category_mat.values
    n_object, n_category = mat.shape
    sum = np.sum(mat, axis=0)
    prob = sum / n_object
    chance = np.matmul(mat, prob)
    chance = np.sum(chance) / n_object
    
    return chance