#%%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import itertools
import optuna
from optuna.storages import RDBStorage


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from sklearn.model_selection import train_test_split, KFold

from joblib import Parallel, delayed

import ot
from sklearn.metrics import pairwise_distances

from src.embedding_model import EmbeddingModel, ModelTraining
from GW_methods.src.align_representations import Representation, VisualizationConfig, AlignRepresentations, OptimizationConfig
from GW_methods.src.utils.utils_functions import get_category_data, sort_matrix_with_categories

#%%

def generate_random_grouping(N_participant, N_groups, seed = 0):
    random.seed(seed)
    
    participants = list(range(N_participant))
    random.shuffle(participants)  # リストをランダムにシャッフルする

    group_size = N_participant // N_groups  # グループごとの理想的なサイズ
    remainder = N_participant % N_groups  # 割り切れなかった場合の余り

    groups = []
    start = 0
    for i in range(N_groups):
        group_end = start + group_size + (1 if i < remainder else 0)  # グループの終了位置
        groups.append(participants[start:group_end])
        start = group_end

    return groups

def make_dataset_COCO(dataset):
    data = torch.LongTensor(dataset.loc[:, 'image1':'image2'].values)
    score = torch.tensor(dataset.loc[:, 'agg_score'].values, dtype=torch.float32)
    dataset = TensorDataset(data, score)

    return dataset
    
class MainTraining():
    def __init__(self, 
                 batch_size, 
                 device, 
                 dataset=None, 
                 train_dataset=None, 
                 valid_dataset=None, 
                 test_size=None) -> None:
        
        self.batch_size = batch_size
        self.device = device
        
        if train_dataset is not None and valid_dataset is not None:
            self.train_dataset = train_dataset
            self.valid_dataset = valid_dataset
            
        else:
            assert dataset is not None and test_size is not None
            self.train_dataset, self.valid_dataset = self._train_test_split_dataset(dataset, test_size)
            

    def _train_test_split_dataset(self, dataset, test_size):
        train_dataset, valid_dataset = train_test_split(dataset, test_size = test_size, shuffle = False, random_state = 42)

        return train_dataset, valid_dataset
    
    def make_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle = True)
        valid_dataloader = DataLoader(self.valid_dataset, self.batch_size, shuffle = False)
        
        return train_dataloader, valid_dataloader
    
    def main_compute(self, loss_fn, emb_dim, object_num, n_epoch, lr, early_stopping=False, distance_metric = "euclidean", lamb = None):
        train_dataloader, valid_dataloader = self.make_dataloader()
        
        model = EmbeddingModel(emb_dim = emb_dim, object_num = object_num).to(self.device)
        
        model_training = ModelTraining(self.device, 
                                       model = model, 
                                       train_dataloader = train_dataloader, 
                                       valid_dataloader = valid_dataloader, 
                                       similarity = "pairwise", 
                                       distance_metric = distance_metric)
        
        test_loss, test_correct = model_training.main_compute(loss_fn=loss_fn, 
                                                lr=lr, 
                                                num_epochs=n_epoch, 
                                                early_stopping=early_stopping,
                                                lamb=lamb, 
                                                show_log=True)
        
        weights = model.state_dict()["Embedding.weight"].to('cpu').detach().numpy().copy()
        
        return weights, test_loss
    
    
class KFoldCV():
    def __init__(self, 
                 dataset, 
                 n_splits, 
                 search_space, 
                 study_name, 
                 results_dir,
                 batch_size,
                     device,
                     loss_fn,
                     emb_dim, 
                     object_num, 
                     n_epoch, 
                     lr, 
                     early_stopping,
                     distance_metric = "euclidean"
                     ) -> None:
        
        self.dataset = dataset
        self.full_data = [data for data in self.dataset]
        self.n_splits = n_splits
        self.search_space = search_space
        
        self.study_name = study_name
        self.save_dir = results_dir + "/" + study_name
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        #self.storage = "sqlite:///" + self.save_dir + "/" + study_name + ".db"
        self.storage = RDBStorage(
            "sqlite:///" + self.save_dir + "/" + study_name + ".db",
            engine_kwargs={"pool_size": 5, "max_overflow": 10},
            #retry_interval_seconds=1,
            #retry_limit=3,
            #retry_deadline=60
            )
        
        self.batch_size = batch_size
        self.device = device
        self.loss_fn = loss_fn
        self.emb_dim = emb_dim
        self.object_num = object_num 
        self.n_epoch = n_epoch
        self.lr = lr
        self.early_stopping = early_stopping
        self.distance_metric = distance_metric
        
    def training(self, trial):
        lamb = trial.suggest_float("lamb", self.search_space[0], self.search_space[1], log=True)
        
        kf = KFold(n_splits=self.n_splits, shuffle=True)
        
        cv_loss = 0
        for train_indices, val_indices in kf.split(self.dataset):
            #train_dataset = Subset(self.dataset, train_indices)
            #valid_dataset = Subset(self.dataset, val_indices)
            train_dataset = [self.full_data[i] for i in train_indices]
            valid_dataset = [self.full_data[i] for i in val_indices]
            valid_dataloader = DataLoader(valid_dataset, self.batch_size, shuffle = False)
            train_dataloader = DataLoader(train_dataset, self.batch_size, shuffle = True)
            
            
            model = EmbeddingModel(emb_dim = self.emb_dim, object_num = self.object_num).to(self.device)
        
            model_training = ModelTraining(self.device, 
                                       model = model, 
                                       train_dataloader = train_dataloader, 
                                       valid_dataloader = valid_dataloader, 
                                       similarity = "pairwise", 
                                       distance_metric = self.distance_metric)
        
            loss, test_correct = model_training.main_compute(loss_fn=self.loss_fn, 
                                                lr=self.lr, 
                                                num_epochs=self.n_epoch, 
                                                early_stopping=self.early_stopping,
                                                lamb=lamb, 
                                                show_log=False)
            
            cv_loss += loss
        cv_loss /= self.n_splits
        
        return cv_loss
    
    def optimize(self, n_trials):
        study = optuna.create_study(study_name = self.study_name, storage = self.storage, load_if_exists = True)
        study.optimize(self.training, n_trials=n_trials)

    def get_best_lamb(self, show_log=False):
        study = optuna.load_study(study_name = self.study_name, storage = self.storage)
        best_trial = study.best_trial
        df_trial = study.trials_dataframe()
        
        if show_log:
            sns.scatterplot(data = df_trial, x = "params_lamb", y = "value", s = 50)
            plt.xlabel("lamb")
            plt.xscale("log")
            plt.ylabel("cv loss")
            fig_path = os.path.join(self.save_dir, f"Optim_log.png")
            plt.savefig(fig_path)
            plt.tight_layout()
            plt.show()
            
        return best_trial.params["lamb"]
        
#%%
if __name__ == "__main__":
    # load the dataset
    category = True
    
    if category:
        data = pd.read_csv('../data/behavior/sis_all_dissim_category.csv', index_col=0)
        image_num = 80
        emb_dim = 50
        save_name = 'embedding_category.npy'
    else:
        data = pd.read_csv('../data/behavior/sis_all_dissim.csv', index_col=0)
        image_num = 10000
        emb_dim = 200
        save_name = 'embedding.npy'
    
    # device 
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Define the optimizer (e.g., Adam)
    lr = 0.001
    num_epochs = 100
    batch_size = 100
    early_stopping = False

    loss_fn = nn.MSELoss()
    distance_metric = "euclidean"
    
    ### cv params
    n_splits = 10
    lamb_range = [1e-3, 1]
    study_name = f"COCO_behavior_metric={distance_metric}{'_category' if category else ''}"
    n_trials = 5

    
    #%%
    ### Main computation
    dataset = make_dataset_COCO(data)
    
    ### cross validation
    cv = KFoldCV(dataset=dataset,
                    n_splits=n_splits,
                    search_space=lamb_range,
                    study_name=study_name,
                    results_dir="../results/cv_embedding_behavior/",
                    batch_size=batch_size,
                    device=device,
                    loss_fn=loss_fn,
                    emb_dim=emb_dim,
                    object_num=image_num,
                    n_epoch=num_epochs,
                    lr=lr,
                    early_stopping=early_stopping,
                    distance_metric=distance_metric)
    
    #cv.optimize(n_trials=n_trials)
    #lamb = cv.get_best_lamb(show_log=True)
    lamb=0.001
    
    ### main
    main_training = MainTraining(dataset = dataset, 
                                    test_size = 1/n_splits, 
                                    batch_size = batch_size, 
                                    device = device)

    embeddings, loss = main_training.main_compute(loss_fn = loss_fn, 
                                emb_dim = emb_dim, 
                                object_num = image_num, 
                                n_epoch = num_epochs, 
                                early_stopping=early_stopping,
                                lr = lr, 
                                distance_metric = distance_metric,
                                lamb=lamb)

    np.save(f"../data/behavior/{save_name}", embeddings)

# %%    
#### check dimensionality
plt.hist(np.abs(embeddings).flatten())
plt.show()

plt.hist(np.max(np.abs(embeddings), axis=0))
plt.show()
#
# %%
### show simmmat
embeddings = np.load(f"../data/behavior/{save_name}")
category_mat = pd.read_csv('../data/behavior/category_mat_behavior.csv', index_col=0)
object_labels, category_idx_list, category_num_list, new_category_name_list = get_category_data(category_mat)

if category:
    behav = Representation(
        name=f"behavior",
        embedding=embeddings,
        metric='euclidean',
        #object_labels=object_labels,
        #category_name_list=new_category_name_list,
        #num_category_list=category_num_list,
        #category_idx_list=category_idx_list,
        #func_for_sort_sim_mat=sort_matrix_with_categories
    )
    
    vis_config = VisualizationConfig(
        cmap='rocket',
    )
    
    behav.show_sim_mat(
        sim_mat_format="default",
        visualization_config=vis_config,
    )

else:
    
    behav = Representation(
            name=f"behavior",
            embedding=embeddings,
            metric='euclidean',
            object_labels=object_labels,
            category_name_list=new_category_name_list,
            num_category_list=category_num_list,
            category_idx_list=category_idx_list,
            func_for_sort_sim_mat=sort_matrix_with_categories
        )

    vis_config = VisualizationConfig(
        cmap='rocket',
        draw_category_line=True, 
        #category_line_color='red', 
        category_line_alpha=0.1, 
        category_line_style='-',
        )
    behav.show_sim_mat(
        sim_mat_format="sorted",
        visualization_config=vis_config,
        ticks='category'
        )
# %%
