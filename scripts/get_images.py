#%%
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%%
with h5py.File('/home1/data/common-data/natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5', 'r') as file:
    # ファイル内のすべてのデータセットをリストアップする
    print(list(file.keys()))
    
    img = file['imgBrick'][1]
    plt.imshow(img)
    plt.axis('off')
    plt.show()
# %%
