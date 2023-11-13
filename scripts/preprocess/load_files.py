#%%
import os
from os.path import join
import nibabel as nib

#%%

sub = 1
data_path = f"/home1/common-data/natural-scenes-dataset/betas/ppdata/subj0{sub}/fsaverage/betas_fithrf_GLMdenoise_RR"

session = "01"
file_name = f"lh.betas_session{session}.mgh"

img = nib.load(join(data_path, file_name))
# %%
data = img.get_fdata()

print(data.shape)
# %%
