#%%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../'))

import pandas as pd

#%%
data_test = pd.read_csv("/home1/user/ken-takeda/Takeda_NSD/Crisscrossed-Captions/data/sis_test.csv")
data_val = pd.read_csv("/home1/user/ken-takeda/Takeda_NSD/Crisscrossed-Captions/data/sis_val.csv")
# %%
# %%
image1 = set(list(data_test['image1'].values))
image2 = set(list(data_test['image2'].values))
image1_val = set(list(data_val['image1'].values))
image2_val = set(list(data_val['image2'].values))

all_image = set(list(data_test['image1'].values) + list(data_test['image2'].values) + list(data_val['image1'].values) + list(data_val['image2'].values))
# %%
