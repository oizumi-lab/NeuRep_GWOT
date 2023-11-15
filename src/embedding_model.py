from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate
import numpy as np
import random
import pandas as pd

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms = False
    
    
class ComputeSimilarityTrip:
    def __init__(self, distance_metric) -> None:
        self.distance_metric = distance_metric
        pass
    
    def cosine_similarity(self, x, T = 20):
        dot_12 = T * F.cosine_similarity(x[:, 0, :], x[:, 1, :], dim = 1).unsqueeze(1)
        dot_23 = T * F.cosine_similarity(x[:, 1, :], x[:, 2, :], dim = 1).unsqueeze(1)
        dot_13 = T * F.cosine_similarity(x[:, 0, :], x[:, 2, :], dim = 1).unsqueeze(1)
        output = torch.cat([dot_23, dot_13, dot_12], dim = 1) #[batch, 3]
        return output
        
    def dot(self, x):
        dot_12 = torch.sum(torch.mul(x[:, 0, :], x[:, 1, :]), dim = 1).unsqueeze(1) #[batch,1]
        dot_23 = torch.sum(torch.mul(x[:, 1, :], x[:, 2, :]), dim = 1).unsqueeze(1)
        dot_13 = torch.sum(torch.mul(x[:, 0, :], x[:, 2, :]), dim = 1).unsqueeze(1)
        output = torch.cat([dot_23, dot_13, dot_12], dim = 1) #[batch, 3]
        return output

    def euclidean(self, x):
        dist_12 = torch.sqrt(torch.norm(x[:, 0, :] - x[:, 1, :], p = 2, dim = 1)).unsqueeze(1)
        dist_23 = torch.sqrt(torch.norm(x[:, 1, :] - x[:, 2, :], p = 2, dim = 1)).unsqueeze(1)
        dist_13 = torch.sqrt(torch.norm(x[:, 0, :] - x[:, 2, :], p = 2, dim = 1)).unsqueeze(1)
        output = torch.cat([-dist_23, -dist_13, -dist_12], dim = 1) #[batch, 3]
        return output

    def __call__(self, x : torch.Tensor, T = 20):
        if self.distance_metric == "cosine":
            return self.cosine_similarity(x, T)
        elif self.distance_metric == "dot":
            return self.dot(x)
        elif self.distance_metric == "euclidean":
            return self.euclidean(x)


class ComputeSimilarityPairwise:
    def __init__(self, distance_metric) -> None:
        self.distance_metric = distance_metric
        pass
    
    def euclidean(self, x):
        ## size of x (batch x 2 x emb_dim)
        # euclidean distance
        sim = torch.norm(x[:, 0, :] - x[:, 1, :], p = 2, dim = 1)
        return sim    
    
    def dot(self, x):
        # dot product
        sim = torch.sum(torch.mul(x[:, 0, :], x[:, 1, :]), dim = 1) #[batch,1]
        return sim
    
    def __call__(self, x : torch.Tensor):
        if self.distance_metric == "dot":
            return self.dot(x)
        elif self.distance_metric == "euclidean":
            return self.euclidean(x)
    
    
class EmbeddingModel(nn.Module):
    def __init__(self, emb_dim, object_num, init_fix = True):
        if init_fix:
            torch_fix_seed()
        super(EmbeddingModel, self).__init__()
        self.Embedding = nn.Embedding(object_num, emb_dim) 
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            #module.weight.data.uniform_(0, 1)
            module.weight.data.normal_(mean=1.0, std=0.1)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x : torch.Tensor):
        x = self.Embedding(x)
        return x
         
    
class ModelTraining():
    def __init__(self, device, model, train_dataloader, valid_dataloader, similarity = "pairwise", distance_metric = "euclidean") -> None:
        self.device = device
        self.model = model
        
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        
        if similarity == "pairwise":
            self.compute_similarity = ComputeSimilarityPairwise(distance_metric)
        elif similarity == "triplet":
            self.compute_similarity = ComputeSimilarityPairwise(distance_metric)
        
    def run_epoch_computation(self, dataloader, loss_fn, optimizer, lamb = None):
        size = len(dataloader.dataset)
        running_loss, correct = 0, 0
        
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            sim = self.compute_similarity(pred)
            loss = loss_fn(sim, y)
            
            if self.model.training == True:
                if lamb is not None:
                    #l1 normalization
                    l1 = 0
                    for w in self.model.parameters():
                        l1 += torch.sum(torch.abs(w))
                    loss += lamb * l1
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item()   
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()  

        running_loss /=  size
        correct /= size
        
        return running_loss, correct
    
    def main_compute(self, loss_fn, lr, num_epochs, early_stopping=True, lamb = None, show_log = True):
        #loss_fn = nn.CrossEntropyLoss(reduction = "sum")
        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        
        training_loss = list()
        testing_loss = list()
        
        best_test_loss = float("inf")
        early_stop_counter = 0
        patience = 5

        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            train_loss, train_correct = self.run_epoch_computation(self.train_dataloader, loss_fn, optimizer, lamb)
            
            with torch.no_grad():
                self.model.eval()
                test_loss, test_correct = self.run_epoch_computation(self.valid_dataloader, loss_fn, optimizer, lamb)
            
            if early_stopping:
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    early_stop_counter = 0
                
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        print("Early stopping triggered. Training stopped.")
                        break
                    
            if show_log: 
                print('[%d/%5d] \ntrain loss: %.5f, accuracy: %.2f \ntest loss: %.5f, accuracy: %.2f' % (epoch + 1, num_epochs, train_loss, train_correct * 100, test_loss, test_correct * 100))
                training_loss.append(train_loss)
                testing_loss.append(test_loss)
        
        if show_log:
            plt.close()
            plt.figure()
            plt.subplot(211)
            plt.plot(training_loss)
            plt.subplot(212)
            plt.plot(testing_loss)
            plt.show()
        
        return test_loss, test_correct

