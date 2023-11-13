#%%
import numpy as np
from sklearn.cluster import KMeans

#%%
category_label = np.load("/home1/data/common-data/natural-scenes-dataset/rsa/all_stims_category_labels.npy", allow_pickle=True)
# %%
unique_categories = set()
for sublist in category_label:
    unique_categories.update(sublist)

unique_categories = list(unique_categories)
np.save("/home1/data/common-data/natural-scenes-dataset/rsa/all_stims_unique_category_labels.npy", unique_categories)

#%%
# 各リストに対して80次元のベクトルを生成
vectors = []
for sublist in category_label:
    # カテゴリリストと同じ長さのゼロベクトルを作成
    vector = np.zeros(len(unique_categories), dtype=int)
    # 各サブリストに含まれるカテゴリのインデックスを取得し、対応するベクトルの要素を1に設定
    for item in sublist:
        index = unique_categories.index(item)  # カテゴリのインデックスを取得
        vector[index] = 1  # 対応する位置を1に設定
    vectors.append(vector)

# 結果のベクトルリスト
vectors = np.array(vectors)

np.save("/home1/data/common-data/natural-scenes-dataset/rsa/all_stims_category_vectors.npy", vectors)

#%%
def find_indices(arr, condition):
    """
    配列 `arr` の中で `condition` 関数に一致する要素のインデックスを返す。

    Parameters:
    arr (np.array): 検索されるNumPy配列
    condition (callable): 各要素に適用される関数。真偽値を返す必要がある。

    Returns:
    np.array: 条件に一致する要素のインデックスの配列
    """
    return np.where(condition(arr))


# KMeansクラスタリングを実行（クラスタ数は適宜選択）
n_clusters = 8  # 5つのクラスタに分ける例
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(vectors)

# クラスタリング結果のラベルを取得
labels = kmeans.labels_

# クラスタのサイズをカウント
cluster_sizes = np.bincount(labels)

# サイズに基づいてクラスタのインデックスをソート（大きい順）
cluster_indices = np.argsort(cluster_sizes)[::-1]

# クラスタをサイズ順に並べ替える
sorted_idx = []
for cluster_index in cluster_indices:
    indices = find_indices(labels, lambda x: x == cluster_index)
    sorted_idx.extend(indices)

#%%
sorted_idx = np.concatenate(sorted_idx)

# 結果の表示
#sorted_idx = np.array(sorted_idx)
print(sorted_idx.shape)

#%%
#check
vectors_sorted = vectors[sorted_idx]

# %%
