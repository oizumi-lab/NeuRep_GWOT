#%%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import optuna
from optuna.storages import RDBStorage
from sklearn.model_selection import train_test_split, KFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the linear model
class LinearRegressionModel(nn.Module):
    def __init__(self, n_category, n_emb):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(n_category, n_emb, bias=False)

    def forward(self, x):
        return self.linear(x)

class ModelTraining():
    def __init__(self, device, model, train_dataloader, valid_dataloader) -> None:
        self.device = device
        self.model = model
        
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        
    def run_epoch_computation(self, dataloader, loss_fn, optimizer, reg=None, lamb = None):
        size = len(dataloader.dataset)
        running_loss, correct = 0, 0
        
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            sim = self.compute_similarity(pred)
            loss = loss_fn(sim, y)
            
            if self.model.training == True:
                if reg == 'l1':
                    #l1 normalization
                    l1 = 0
                    for w in self.model.parameters():
                        l1 += torch.norm(w, 1)
                    loss += lamb * l1
                    
                elif reg == 'l2':
                    #l1 normalization
                    l2 = 0
                    for w in self.model.parameters():
                        l2 += torch.norm(w)**2
                    loss += lamb * l2
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item()   
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()  

        running_loss /=  size
        correct /= size
        
        return running_loss, correct
    
    def main_compute(
        self, 
        loss_fn, 
        optimizer, 
        num_epochs, 
        early_stopping=True, 
        reg=None,
        lamb = None, 
        show_log = True
        ):
        #loss_fn = nn.CrossEntropyLoss(reduction = "sum")
        #optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        
        training_loss = list()
        testing_loss = list()
        
        best_test_loss = float("inf")
        early_stop_counter = 0
        patience = 5

        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            train_loss, train_correct = self.run_epoch_computation(self.train_dataloader, loss_fn, optimizer, reg, lamb)
            
            with torch.no_grad():
                self.model.eval()
                test_loss, test_correct = self.run_epoch_computation(self.valid_dataloader, loss_fn, optimizer,reg, lamb)
            
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
    
    def main_compute(self, loss_fn, n_category, n_emb, n_epoch, lr, early_stopping=False, reg=None, lamb = None):
        train_dataloader, valid_dataloader = self.make_dataloader()
        
        model = LinearRegressionModel(n_category = n_category, n_emb = n_emb).to(self.device)
        
        model_training = ModelTraining(self.device, 
                                       model = model, 
                                       train_dataloader = train_dataloader, 
                                       valid_dataloader = valid_dataloader, 
                                       )
        
        test_loss, test_correct = model_training.main_compute(loss_fn=loss_fn, 
                                                lr=lr, 
                                                num_epochs=n_epoch, 
                                                early_stopping=early_stopping,
                                                reg=reg,
                                                lamb=lamb, 
                                                show_log=True)
        
        weights = model.state_dict()["Embedding.weight"].to('cpu').detach().numpy().copy()
        
        return weights, test_loss
    

class KFoldCV():
    def __init__(
        self, 
        dataset, 
        n_splits, 
        search_space, 
        study_name, 
        results_dir,
        batch_size,
        device,
        loss_fn,
        n_category, 
        n_emb, 
        n_epoch, 
        lr, 
        reg,
        early_stopping,
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
        self.n_category = n_category
        self.n_emb = n_emb
        self.n_epoch = n_epoch
        self.lr = lr
        self.reg = reg
        self.early_stopping = early_stopping
        
        
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
            
            
            model = LinearRegressionModel(n_category = self.n_category, n_emb = self.n_emb).to(self.device)
        
            model_training = ModelTraining(self.device, 
                                       model = model, 
                                       train_dataloader = train_dataloader, 
                                       valid_dataloader = valid_dataloader, 
                                       )
        
            loss, test_correct = model_training.main_compute(loss_fn=self.loss_fn, 
                                                lr=self.lr, 
                                                num_epochs=self.n_epoch, 
                                                early_stopping=self.early_stopping,
                                                reg=self.reg,
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
if __name__ == '__main__':
    # Assuming n_vox, n_stim, and n_category are defined
    n_emb = 10000  # Example large size
    n_stim = 20000  # Example large size
    n_category = 1000  # Example large size

    # Check for GPU
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


    # Instantiate the model and move it to GPU if available
    model = LinearRegressionModel(n_category, n_emb).to(device)

    # Loss and Optimizer with L2 Regularization
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Using Adam optimizer

    # Assuming B and D are given as torch tensors
    # B size: (n_emb, n_stim)
    
    # D size: (n_stim, n_category)
    Betas = torch.randn(n_stim, n_emb).to(device)  # Move data to GPU
    Design_mat = torch.randn(n_stim, n_category).to(device)  # Move data to GPU

    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(Design_mat)
        loss = criterion(outputs, Betas)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

    #%%
    # After training, model.linear.weight will contain the learned weights W
    weights = model.state_dict()["linear.weight"].to('cpu').detach().numpy().copy()


# %%
