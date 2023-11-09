#%%
import os
from os.path import join

import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
from tqdm import tqdm

#%%
data_path = '/home1/data/common-data/natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/'

with h5py.File(join(data_path, 'nsd_stimuli.hdf5'), 'r') as file:
    # ファイル内のすべてのデータセットをリストアップする
    print(list(file.keys()))
    img_file = file['imgBrick']
    print(len(img_file))
    
    for i in tqdm(range(len(img_file))):
        img = img_file[i]
        img = np.array(img)
        img = Image.fromarray(img.astype('uint8'), 'RGB')

        os.makedirs(join(data_path, f'image{i}/'), exist_ok=True)
        img.save(join(data_path, f'image{i}/image{i}.png'))
# %%
