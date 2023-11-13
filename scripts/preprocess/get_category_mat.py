#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from collections import defaultdict
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')

#%%
def convert_to_max_zero_columns(matrix):
    n, m = matrix.shape

    # Step 1: Convert rows to one-hot vectors
    for i in tqdm(range(n)):
        if np.sum(matrix[i]) > 1:  # If the row is not already a one-hot vector
            # Find the column with the maximum number of '1's
            max_ones_col = np.argmax(np.sum(matrix, axis=0) * matrix[i])
            # Set all but the selected '1' to '0'
            matrix[i] = np.where(np.arange(m) == max_ones_col, matrix[i], 0)

    # Step 2: Set columns to zero vectors if possible
    for col in tqdm(range(m)):
        if np.sum(matrix[:, col]) == 0:
            matrix[:, col] = 0

    return matrix

def categorize_labels(labels):
    """
    Categorize the labels into broader categories based on their WordNet hypernyms.
    """
    categories = defaultdict(list)

    for label in labels:
        # Get synsets for the label
        synsets = wn.synsets(label)
        if synsets:
            # Use the first synset
            hypernyms = synsets[0].hypernyms()
            if hypernyms:
                # Use the first hypernym's name as the category
                category = hypernyms[0].name().split('.')[0]
                categories[category].append(label)
            else:
                # If no hypernyms, categorize as 'miscellaneous'
                categories['miscellaneous'].append(label)
        else:
            # If no synsets found, categorize as 'unknown'
            categories['unknown'].append(label)

    return categories
#%%
if __name__ == '__main__':
    category_label = np.load("/home1/data/common-data/natural-scenes-dataset/rsa/all_stims_category_labels.npy", allow_pickle=True)
    category_vectors = np.load("/home1/data/common-data/natural-scenes-dataset/rsa/all_stims_category_vectors.npy")
    unique_categories = np.load("/home1/data/common-data/natural-scenes-dataset/rsa/all_stims_unique_category_labels.npy")
    
    # Show the histogram of how many annotations each image has
    category_summary = np.sum(category_vectors, axis=0)
    print(category_summary.shape)
    # %%
    plt.figure()
    plt.hist(category_summary)
    plt.show()

    print("min : ", np.min(category_summary))
    print("max : ", np.max(category_summary))
    # %%
    print("most major label : ", unique_categories[np.argmax(category_summary)])
    #%%

    # Example matrix
    matrix = np.array([[1, 0, 1, 0, 1, 0],
                       [0, 0, 1, 0, 0, 0],
                       [1, 0, 0, 0, 0, 1],
                       [0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 1, 0]])

    matrix = np.array([[1, 0, 0, 0, 0, 1],
                       [1, 0, 0, 0, 0, 1],
                       [1, 0, 0, 0, 0, 1],
                       [1, 0, 0, 0, 0, 1],
                       [1, 0, 0, 0, 0, 1],
                       [1, 0, 0, 0, 0, 1]])

    result = convert_to_max_zero_columns(matrix)
    print(result)

    # %%
    category_mat = convert_to_max_zero_columns(category_vectors)
    # %%
    conditions = np.load('/home1/data/common-data/natural-scenes-dataset/rsa/all_subs_stims_full.npy', allow_pickle=True)
    conditions -= 1
    #%%
    category_mat_df = pd.DataFrame(data=category_mat, columns=unique_categories, index=conditions)
    category_mat_df.to_csv("../data/category_mat.csv")

    # %%
    #sort labels
    # Categorize the labels
    #categorized_labels = categorize_labels(unique_categories)

    #%%
    # Display the categorized labels (first few categories for brevity)
    #list(categorized_labels.items())[:5]

    #%%
    # manual
    sorted_list = [
        'person',
        'elephant', 'zebra', 'sheep', 'mouse', 'horse', 'dog', 'cat', 'cow', 'bear', 'giraffe', 'bird', 'teddy bear',
        'bicycle', 'motorcycle', 'bus', 'car', 'airplane', 'boat', 'train', 'truck',
        'spoon', 'knife', 'fork', 'bowl', 'cup', 'bottle', 'wine glass',
        'banana', 'carrot', 'apple', 'orange', 'cake', 'pizza', 'sandwich', 'hot dog', 'donut', 'broccoli',
        'bench', 'dining table', 'chair', 'couch', 'bed', 'oven', 'toaster', 'microwave', 'sink', 'refrigerator', 'toilet', 'potted plant', 'vase',
        'laptop', 'cell phone', 'tv', 'remote', 'keyboard', 'clock', 'hair drier',
        'baseball bat', 'skateboard', 'skis', 'snowboard', 'sports ball', 'tennis racket', 'frisbee', 'surfboard', 'baseball glove',
        'backpack', 'handbag', 'suitcase', 'tie', 'toothbrush', 'umbrella', 
        'parking meter', 'kite', 'book', 'traffic light', 'stop sign', 'fire hydrant', 'scissors'
    ]
    
    set_org = set(list(unique_categories))
    set_sorted = set(sorted_list)

    print("missing in sorted list : ", list(set_org - set_sorted))
    print("elements that the original doesn't have : ", list(set_sorted - set_org))
    # %%
    category_mat_df_sorted = category_mat_df.reindex(columns=sorted_list)
    category_mat_df_sorted.to_csv("../data/category_mat.csv")
    # %%
    # extract shared 515
    shared515 = np.array(np.load("../data/shared515ids.npy"))
    category_mat_shared515 = category_mat_df_sorted.loc[shared515-1]
    category_mat_shared515.to_csv('../data/category_mat_shared515.csv')
    # %%
    # save original
    original_mat = pd.DataFrame(data=category_vectors, columns=unique_categories, index=conditions)
    original_mat = original_mat.reindex(columns=sorted_list)
    original_mat.to_csv("../data/category_mat_original.csv")
# %%
