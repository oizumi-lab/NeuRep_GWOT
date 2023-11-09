#%%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.load_img import GWD_Dataset
#%%
save_path = '../data/shared_515'
dataset_path = '/home1/data/common-data/natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/shared_515'

model_list = [
    'AlexNet', 
    'VGG19',
    'CLIP_B16_OpenAI',
    'CLIP_B16_datacomp_l_s1b_b8k',
    'CLIP_B16_datacomp_xl_s13b_b90k',
    'CLIP_B16_laion2B-s34B-b88K', 
    # 'CLIP_L14_commonpool_xl_laion_s13b_b90k', 
    'ViT_B16_ImageNet1K', 
    'ViT_B16_ImageNet21K',
]

for model_name in model_list:
    test = GWD_Dataset(model_name, save_path, dataset_path)
    cosine_dis_sim, label = test.extract()

# %%
