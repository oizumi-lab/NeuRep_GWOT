#%%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import pandas as pd
import numpy as np
import json

#%%
## load category mat
category_mat = pd.read_csv('../data/behavior/category_mat_behavior.csv', index_col=0).values.astype('float64')
category_mat /= (np.sum(category_mat, axis=0).reshape(1, -1) + 1e-10)

### load embeddings
embeddings = np.load('../data/behavior/embeddings.npy')
# %%
avg_emb = np.matmul(category_mat.T, embeddings)
# %%
np.save('../data/behavior/category_embeddings', avg_emb)
# %%
