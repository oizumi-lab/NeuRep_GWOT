#%%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../'))

import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from cocoapi.PythonAPI.pycocotools.coco import COCO

#%%
### load behavioral data
data_test = pd.read_csv("/home1/user/ken-takeda/Takeda_NSD/Crisscrossed-Captions/data/sis_test.csv")
data_val = pd.read_csv("/home1/user/ken-takeda/Takeda_NSD/Crisscrossed-Captions/data/sis_val.csv")
# %%
image1_test = list(data_test['image1'].values)
image2_test = list(data_test['image2'].values)
image1_val = list(data_val['image1'].values)
image2_val = list(data_val['image2'].values)

images_test = list(set(image1_test + image2_test))
images_val = list(set(image1_val + image2_val))

all_image = set(list(data_test['image1'].values) + list(data_test['image2'].values) + list(data_val['image1'].values) + list(data_val['image2'].values))
all_image = list(all_image)

# save all image names
with open('../data/behavior/behavior_all_images.json', 'w') as f:
    json.dump(all_image, f)

#%%
### convert each image names to ids
def list_to_dict(lst):
    return {element: index for index, element in enumerate(lst)}

id_dict = list_to_dict(all_image)    

### replace original dataframe
data_test = data_test.replace(id_dict)
data_val = data_val.replace(id_dict)

# concatenate them
data_all = pd.concat([data_test, data_val])

#%%
data_dissim = data_all
data_dissim['agg_score'] = 5 - data_all['agg_score']

# save
data_all.to_csv('../data/behavior/sis_all.csv')
data_dissim.to_csv('../data/behavior/sis_all_dissim.csv')
#%%

### load annotation file
dataDir='/home1/data/common-data/COCO/annotations/'
dataType='val2014'
annFile=f'{dataDir}/annotations/instances_{dataType}.json'

# initialize COCO api for instance annotations
coco=COCO(annFile)

#%%
### read category info
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]

# Write JSON to a file
with open('../data/categories.json', 'w') as f:
    json.dump(cats, f)

# %%
### read the original file
with open(annFile, 'r') as f:
    dataset = json.load(f)
# %%
def extract_ImgIds(dataset, files):
    imgs = dataset['images']
    all_file_name = [img['file_name'] for img in imgs]
    all_ids = [img['id'] for img in imgs]
    
    Img_indices = [all_file_name.index(file) for file in files]
    ImgIds = [all_ids[id] for id in Img_indices]
    
    return ImgIds

def ImgIds_to_CatIds(dataset, ImgIds):
    annotations = dataset['annotations']
    all_ImgIds = [ann['image_id'] for ann in annotations]
    all_CatIds = [ann['category_id'] for ann in annotations]
    
    CatIds = []
    Nones = 0
    for id in tqdm(ImgIds):
        if id in all_ImgIds:
            idx = all_ImgIds.index(id)
            CatIds.append(all_CatIds[idx])
            
        else:
            CatIds.append('None')
            Nones += 1

    return CatIds, Nones

def CatIds_to_CatNames(categories, CatIds):
    cats = [cat['id'] for cat in categories]
    names = [cat['name'] for cat in categories]
    
    CatNames = []
    CatVecs = []
    for CatId in CatIds:
        CatVec = np.zeros((80,))
        if CatId == 'None':
            CatNames.append('None')
            CatVecs.append(CatVec)
            
        else:
            idx = cats.index(CatId)
            CatNames.append(names[idx])
            
            CatVec[idx] = 1
            CatVecs.append(CatVec)
            
    return CatNames, CatVecs
    
# %%
ImgIds = extract_ImgIds(dataset, all_image)
#%%
CatIds, Nones = ImgIds_to_CatIds(dataset, ImgIds)
# %%
CatNames, CatVecs = CatIds_to_CatNames(cats, CatIds)
# %%
CatVecs = np.array(CatVecs).astype(int)
category_mat = pd.DataFrame(data = CatVecs, index=all_image, columns=nms)
# %%
category_mat.to_csv('../data/category_mat_behavior.csv')
# %%
