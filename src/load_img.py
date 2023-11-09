# %%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# %%
import gc
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision, torchmetrics
from torchvision import transforms
from tqdm.auto import tqdm
import torch
import open_clip
#from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import transformers

# test_model = torchvision.models.vit_b_16(weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)#.to(self.device)
# test_img = torch.randn(32,3,224,224)
# test_out = test_model(test_img)

# %%
# ImageBind
# from ImageBind.models import imagebind_model
# from ImageBind.models.imagebind_model import ModalityType
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTImageProcessor, ViTModel

# %%
# nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

THINGS_IMAGE_PATH = "/home/masaru-sasaki/Data/THINGS/Images"
THINGS_IMAGE_METADATA_PATH = "/home1/common-data/THINGS-images/THINGS/Metadata"
THINGS_CONCEPTS_IMAGE_PATH = "/home1/common-data/THINGS-images/concepts/" #61番用

NSD_IMAGE_PATH = "/home1/data/common-data/natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

GOOGLENET_MEAN = [0.5, 0.5, 0.5]
GOOGLENET_STD = [0.5, 0.5, 0.5]

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

BATCH_SIZE = 10
PIC_SIZE = 256 #256
CROP_SIZE = 224 #224

# %% 
def transform_to_show(img):
    check_pic = img.to('cpu').mul(torch.FloatTensor(IMAGENET_STD).view(3, 1, 1))
    check_pic = check_pic.add(torch.FloatTensor(IMAGENET_MEAN).view(3, 1, 1))
    ori_check = transforms.functional.to_pil_image(check_pic)
    return ori_check

def torch_fix_seed(seed=42):
    import random
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


# 使用データの定義
class GWD_Dataset():
    def __init__(self, model_name, save_path, dataset_path, target="conv", device = 'cuda:0') -> None:
        self.model_name = model_name
        self.device = device
        self.target = target
        self.save_path = save_path
        self.emb_path = os.path.join(self.save_path, 'emb', self.model_name.lower() + '_' + self.target + '.pt')
        self.sim_mat_path = os.path.join(self.save_path, 'sim_mat', self.model_name.lower() + '_' + self.target + '.pt')
        self.all_images_path = os.path.join(self.save_path, 'all_images', self.model_name.lower() + '_' + self.target + '.pt')
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            os.makedirs(os.path.join(self.save_path, 'emb'))
            os.makedirs(os.path.join(self.save_path, 'sim_mat'))
        
        # 各モデルの定義    
        if 'vit_b16_imagenet1k' in self.model_name.lower():
            self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').eval().to(self.device)
        
        elif 'vit_b16_imagenet21k' in self.model_name.lower():
            self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').eval().to(self.device)
                    
        elif 'alexnet' in self.model_name.lower():
            self.model = torchvision.models.alexnet(weights="IMAGENET1K_V1").eval().to(self.device)
            
        elif 'vgg19' in self.model_name.lower():
            self.model = torchvision.models.vgg19(weights="IMAGENET1K_V1").eval().to(self.device)
        
        # elif 'imagebind' in self.model_name.lower():
        #     self.model = imagebind_model.imagebind_huge(pretrained=True).eval().to(device)
        
        # CLIP_B16
        elif 'clip_b16_openai' in self.model_name.lower():
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
            self.model = model.eval().to(self.device)
            
        elif 'clip_b16_datacomp_l_s1b_b8k' in self.model_name.lower():
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='datacomp_l_s1b_b8k')
            self.model = model.eval().to(self.device)
            
        elif 'clip_b16_datacomp_xl_s13b_b90k' in self.model_name.lower():
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='datacomp_xl_s13b_b90k')
            self.model = model.eval().to(self.device)
        
        elif 'clip_b16_commonpool_l_s1b_b8k' in self.model_name.lower():
            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-CommonPool.L-s1B-b8K')
            self.model = model.eval().to(self.device)
        
        elif 'clip_b16_laion2b-s34b-b88k' in self.model_name.lower():
            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
            self.model = model.eval().to(self.device)
            
        
        #CLIP_B32
        elif 'clip_b32_openai' in self.model_name.lower():
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
            self.model = model.eval().to(self.device)
        
        elif 'clip_b32-laion2b-s34b-b79k' in self.model_name.lower():
            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
            self.model = model.eval().to(self.device)
        
        elif 'CLIP_B32_datacomp_xl_s13b_b90k' in self.model_name:
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='datacomp_xl_s13b_b90k')
            self.model = model.eval().to(self.device)
        
        # CLIP_L14
        elif 'clip_l14_openai' in self.model_name.lower():
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
            self.model = model.eval().to(self.device)
      
        elif 'clip_l14_commonpool_xl_laion_s13b_b90k' in self.model_name.lower():
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='commonpool_xl_laion_s13b_b90k')
            self.model = model.eval().to(self.device)
        
        elif 'clip_l14_datacomp_xl_s13b_b90k' in self.model_name.lower():
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')
            self.model = model.eval().to(self.device)
            
        elif 'clip_l14_laion2b_s32b_b82k' in self.model_name.lower():
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
            self.model = model.eval().to(self.device)
    
        # elif 'clip_h14-laion2b-s32b-b79k' in self.model_name.lower():
        #     model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
        #     self.model = model.eval().to(self.device)    
        
        # elif 'clip_bigg14-laion2b-39b-b160k' in self.model_name.lower():    
        #     model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
        #     self.model = model.eval().to(self.device)
        
        # elif 'clip_bigg-14-clipa-datacomp1b' in self.model_name.lower():
        #     model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:rwightman/ViT-bigG-14-CLIPA-datacomp1B')

        else:
            Exception('model_name is not correct.')
                            
        for params in self.model.parameters():
            params.requires_grad = False
        
        if 'vit' in self.model_name.lower():
            pre_mean = GOOGLENET_MEAN
            pre_std = GOOGLENET_STD
        
        else:
            pre_mean = IMAGENET_MEAN
            pre_std = IMAGENET_STD
        
        transform = transforms.Compose([
                        transforms.Resize(PIC_SIZE), # (256, 256) に変換。
                        transforms.CenterCrop(CROP_SIZE),  # 画像の中心に合わせて、(224, 224) で切り抜く
                        transforms.ToTensor(),  # テンソルにする。
                        transforms.Normalize(pre_mean, pre_std),  # 標準化する。
                    ])

        self.dataset = torchvision.datasets.ImageFolder(root = dataset_path, transform = transform)

            
    def _extract_feat_from_dataset(self, dataset_now):
        """
        # GWDの計算に使うデータを取得していく。

        Args:
            dataset_now (torch.utils.data.Dataset) : 計算に使うデータセット
        Returns:
            feats : modelの特徴量
        """
        feats = []
        label = []
        
        loader = DataLoader(dataset_now, batch_size = 32, num_workers = 8, shuffle = False)

        for img, lab in tqdm(loader):
            feat1 = self._extract_latent(img.to(self.device))
            
            feat1 = feat1.reshape(len(feat1), -1)
            
            feats.append(feat1)
            label.append(lab)

        feats = torch.cat(feats)
        label = torch.cat(label)

        return feats, label


    def _extract_latent(self, img):
        with torch.no_grad():
            if 'imagenet1k' in self.model_name.lower():
                return_dict = self.model.config.use_return_dict

                outputs = self.model.vit(
                    pixel_values = img,
                    head_mask=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    interpolate_pos_encoding=None,
                    return_dict=return_dict,
                )

                batch_feat = outputs[0][:,0,:]
            
            elif 'imagenet21k' in self.model_name.lower():
                outputs = self.model(img)
                last_hidden_state = outputs.last_hidden_state.detach()
                batch_feat = last_hidden_state[:,0,:]
                
            elif 'alexnet' in self.model_name.lower():
                if self.target == 'conv':
                    batch_feat = self.model.features(img).detach()  
            
            elif 'vgg19' in self.model_name.lower():
                if self.target == 'conv':
                    batch_feat = self.model.features(img).detach()

            elif 'clip' in self.model_name.lower():
                # 'ViT-L-14', 'commonpool_xl_laion_s13b_b90k'
                batch_feat = self.model.encode_image(img)
                
            # elif 'imagebind' in self.model_name.lower():
            
            #     inputs = {ModalityType.VISION:img}
                
            #     embeddings = self.model(inputs)

            #     batch_feat = embeddings[ModalityType.VISION]
      
            else:
                raise Exception('model_name is not correct.')
            
        torch.cuda.empty_cache()
        gc.collect()
        return batch_feat

    def extract(self):
        # 計算に使うデータを取得。
        if not os.path.exists(self.emb_path):
            feats, label = self._extract_feat_from_dataset(self.dataset)
            torch.save({'feats' : feats.to('cpu'), 'label' : label.to('cpu')}, self.emb_path)
        else:
            tt = torch.load(self.emb_path)
            feats = tt['feats']
            label = tt['label']
        
        model_cos_torch = 1 - torchmetrics.functional.pairwise_cosine_similarity(feats, feats)
        self.check_dataset(model_cos_torch)
        
        torch.save(model_cos_torch.to('cpu'), self.sim_mat_path)
            
        return model_cos_torch, label  

    def check_dataset(self, model):
        plt.figure()
        plt.title(self.model_name)
        plt.imshow(model.clone().to('cpu').numpy(), cmap=plt.cm.jet)
        plt.colorbar()
        plt.show()


    def make_avg_class_from_all_images(self, feats, label):
        if not os.path.exists(self.all_images_path):
            classes, num_item_each_label = torch.unique(label, return_counts = True) 

            mean_cos_sim = []
            for lab in tqdm(classes):
                cos_sim = feats[label == lab]

                tt = torch.split(cos_sim, num_item_each_label.tolist(), dim=1)
                mean_tt = torch.tensor([torch.mean(a) for a in tt])
                mean_cos_sim.append(mean_tt)
                
            mean_cos_sim = torch.cat(mean_cos_sim).reshape(len(mean_cos_sim), *mean_cos_sim[0].shape)
            torch.save(mean_cos_sim, self.all_images_path)
        
        else:
            mean_cos_sim = torch.load(self.all_images_path)
        
        self.check_dataset(mean_cos_sim)
        
        return mean_cos_sim
    

# %%
if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    # model_list = [
    #     'AlexNet', 
    #     'VGG19',
    #     'ImageBind', 
    #     'ViT_B16_ImageNet1K', 
    #     'ViT_B16_ImageNet21K',
    # ]
    
    b16_list = [
        'CLIP_B16_OpenAI',
        'CLIP_B16_datacomp_l_s1b_b8k',
        'CLIP_B16_datacomp_xl_s13b_b90k',
        'CLIP_B16_laion2B-s34B-b88K',
        'CLIP_B16_commonpool_l_s1b_b8k',
    ]
    
    #b16_additional_list = [
        # 'commonpool_l_clip_s1b_b8k', 
        # 'commonpool_l_laion_s1b_b8k', 
        # 'commonpool_l_image_s1b_b8k', 
        # 'commonpool_l_text_s1b_b8k', 
        # 'commonpool_l_basic_s1b_b8k', 
        # 'commonpool_l_s1b_b8k'
        # ]
    
    b32_list = [
        'CLIP_B32_OpenAI',
        # 'CLIP_B32-laion2B-s34B-b79K',
        'CLIP_B32_datacomp_xl_s13b_b90k',
    ]
    
    # b32_list_additional = [
    #     'openai', 
    #     'laion400m_e31', 
    #     'laion400m_e32', 
    #     'laion2b_e16', 
    #     'laion2b_s34b_b79k', 
    #     'datacomp_xl_s13b_b90k', 
    #     'datacomp_m_s128m_b4k', 
    #     'commonpool_m_clip_s128m_b4k', 
    #     'commonpool_m_laion_s128m_b4k', 
    #     'commonpool_m_image_s128m_b4k', 
    #     'commonpool_m_text_s128m_b4k', 
    #     'commonpool_m_basic_s128m_b4k', 
    #     'commonpool_m_s128m_b4k', 
    #     'datacomp_s_s13m_b4k', 
    #     'commonpool_s_clip_s13m_b4k', 
    #     'commonpool_s_laion_s13m_b4k', 
    #     'commonpool_s_image_s13m_b4k', 
    #     'commonpool_s_text_s13m_b4k', 
    #     'commonpool_s_basic_s13m_b4k', 
    #     'commonpool_s_s13m_b4k'
    # ]
        
    
    l14_list = [
        'CLIP_L14_OpenAI',
        'CLIP_L14_laion2b_s32b_b82k', 
        'CLIP_L14_datacomp_xl_s13b_b90k',
        # 'CLIP_L14_commonpool_xl_laion_s13b_b90k',
    ]
    
    # l14_list = [
        # 'openai', 
        # 'laion400m_e31', 
        # 'laion400m_e32', 
        # 'laion2b_s32b_b82k', 
        # 'datacomp_xl_s13b_b90k', 
        # 'commonpool_xl_clip_s13b_b90k', 
        # 'commonpool_xl_laion_s13b_b90k', 
        # 'commonpool_xl_s13b_b90k']
    
    h14_list = ['CLIP_L14-laion2B-s32B-b79K']
    
    g14_list = [
        'CLIP_bigG14-laion2B-39B-b160k',
        'CLIP_bigG-14-CLIPA-datacomp1B',
    ]
    
    save_path = '../data/concepts/'
    
    
    # %%
    model_list = l14_list
    for model_name in model_list:
        test = GWD_Dataset(model_name, save_path)
        cosine_dis_sim, label = test.extract()
    
    # %%
    # mean_cos_sim = test.make_avg_class_from_all_images(cosine_dis_sim, label)
    
# %%
