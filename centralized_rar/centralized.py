import wfdb
import torch
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torchaudio.functional import lowpass_biquad
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn as nn
import os
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DatasetECG(Dataset):
    def __init__(self,samples,labels):
        self.samples = samples 
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        sample = self.samples[idx]
        sample = lowpass_biquad(sample, 360, 60, 0.7)
#         sample = (sample-sample.mean())/sample.std()
#         sample = (sample-sample.min())/(sample.max()-sample.min())
        sample = (sample-sample.min())/(sample.max()-sample.min())

        return sample, label
		

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity = 'relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
		

def load_data():
    with open(r"/home/favaro/TCCLucas/TCC/centralized_rar/training_seq0", "rb") as fp:
        training_sequences = pickle.load(fp)

    with open(r"/home/favaro/TCCLucas/TCC/centralized_rar/training_lab0", "rb") as fp:
        training_labels = pickle.load(fp)

    with open(r"/home/favaro/TCCLucas/TCC/testing_seq", "rb") as fp:
        test_sequences = pickle.load(fp)

    with open(r"/home/favaro/TCCLucas/TCC/testing_lab", "rb") as fp:
        test_labels = pickle.load(fp)

    with open(r"/home/favaro/TCCLucas/TCC/val_sequences", "rb") as fp:
        val_sequences = pickle.load(fp)

    with open(r"/home/favaro/TCCLucas/TCC/val_labels", "rb") as fp:
        val_labels = pickle.load(fp)    
        
    train_normal_len = training_labels.count(0)
    train_s_len = training_labels.count(1)
    train_v_len = training_labels.count(2)
    train_f_len = training_labels.count(3)
    train_q_len = training_labels.count(4)
    t_normal_len = test_labels.count(0)
    t_s_len = test_labels.count(1)
    t_v_len = test_labels.count(2)
    t_f_len = test_labels.count(3)
    t_q_len = test_labels.count(4)
    print("Train: ",train_normal_len,train_s_len,train_v_len,train_f_len,train_q_len)
    print("Test: ",t_normal_len,t_s_len,t_v_len,t_f_len,t_q_len)

    train_ab_lab = [train_s_len,train_v_len,train_f_len,train_q_len]
    train_ab = [[i for n,i in enumerate(training_sequences) if training_labels[n]==1],
               [i for n,i in enumerate(training_sequences) if training_labels[n]==2],
               [i for n,i in enumerate(training_sequences) if training_labels[n]==3],
                [i for n,i in enumerate(training_sequences) if training_labels[n]==4],
               ]

    counter = 0
    for i in range(1,5,1):
        while (training_labels.count(i)+1)<(8.9*train_ab_lab[i-1]):
            index = np.random.randint(0,len(train_ab[i-1]))
            training_sequences.append(train_ab[i-1][index])
            training_labels.append(i)
        print(i)
        
    training_labels = [i if i == 0 else 1 for i in training_labels]
    test_labels = [i if i == 0 else 1 for i in test_labels]
    val_labels = [i if i == 0 else 1 for i in val_labels]
    print(len(test_labels))
    train_dataset = DatasetECG(training_sequences,training_labels)
    labels_counts, counts = np.unique(training_labels, return_counts=True)
    class_weights = [sum(counts)/c for c in counts]

    train_samples_weight = [class_weights[class_id] for class_id in training_labels]
    train_sampler = torch.utils.data.WeightedRandomSampler(train_samples_weight, len(train_samples_weight), replacement=True)

    trainloader = DataLoader(train_dataset,batch_size=100,
                             shuffle=True,
    #                          sampler=train_sampler
                            )
    
    val_dataset = DatasetECG(val_sequences,val_labels)
    val_loader = DataLoader(val_dataset,batch_size=1,shuffle=True)
    
    test_dataset = DatasetECG(test_sequences,test_labels)
    testloader = DataLoader(test_dataset,batch_size=1)
    print(len(test_dataset))
    return trainloader, val_loader, testloader, test_labels
	

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=3)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8,kernel_size=21)
        self.pool = nn.MaxPool1d(2)
        self.pool2 = nn.AvgPool1d(2)
        self.norm1 = nn.BatchNorm1d(8,momentum=0.1)
        self.norm2 = nn.BatchNorm1d(8,momentum=0.1)
        self.norm3 = nn.BatchNorm1d(8)
        self.norm4 = nn.BatchNorm1d(4)
        self.norm5 = nn.BatchNorm1d(2)

        self.drop = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=21)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=8,kernel_size=21)

        self.fc1 = nn.Linear(1320, 1320)
        self.fc2 = nn.Linear(1320, 1)
        self.fc3 = nn.Linear(9, 1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.drop(x)
        x = F.leaky_relu(x)
        x = self.pool(x)

        
        x = self.conv2(x)
        x = self.drop(x)
        x = F.leaky_relu(x)
        x = self.pool(x)


        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.drop(x)

        x = self.fc2(x)


        return x
		

def train(net, trainloader, val_loader):
    np.random.seed(42)
    torch.manual_seed(42)
    os.environ["PYTHONHASHSEED"] = "42"
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    
    net.apply(init_weights)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9,weight_decay=0.001)


    list_loss_train = []
    list_loss_test = []
    for epoch in range(50):
        epoch_loss_training = 0
        epoch_acc_training = 0

        epoch_loss_testing = 0
        epoch_acc_testing = 0
        #
        for training_val in [0,1]:
        #

            if training_val == 0:
                net.train()
                for i, data in enumerate(trainloader):
                    inputs, label = data
                    inputs = inputs.to(device)
                    label = label.to(device)
                    optimizer.zero_grad()
                    outputs = net(inputs.unsqueeze(1))
                    loss = criterion(outputs, label.unsqueeze(1).to(torch.float32))
                    acc = binary_acc(outputs, label.unsqueeze(1).to(torch.float32))
                    
                    l1_crit = nn.L1Loss(size_average=False)
                    reg_loss = 0
                    for param in net.parameters():
                        reg_loss += l1_crit(param,target=torch.zeros_like(param))

                    factor = 0.001
                    loss += factor * reg_loss
                    
                    loss.backward()
                    optimizer.step()

                    epoch_loss_training += loss.item()
                    epoch_acc_training += acc.item()
            else:
                net.eval()
                with torch.no_grad():
                    for item in val_loader:
                        X_batch, label = item
                        X_batch = X_batch.to(device)
                        label = label.to(device)
                        y_test_pred = net(X_batch.unsqueeze(1))
                        loss = criterion(y_test_pred, label.unsqueeze(1).to(torch.float32))
                        acc = binary_acc(y_test_pred, label.unsqueeze(1).to(torch.float32))

                        l1_crit = nn.L1Loss(size_average=False)
                        reg_loss = 0
                        for param in net.parameters():
                            reg_loss += l1_crit(param,target=torch.zeros_like(param))

                        factor = 0.001
                        loss += factor * reg_loss

                        epoch_loss_testing += loss.item()
                        epoch_acc_testing+= acc.item()

        list_loss_train.append(epoch_loss_training)
        list_loss_test.append(epoch_loss_testing)
        print(f'Epoch {epoch+0:03}: | Loss: {epoch_loss_training/len(trainloader):.5f} | {epoch_loss_testing/len(val_loader):.5f} | Acc: {epoch_acc_training/len(trainloader):.3f}  | {epoch_acc_testing/len(val_loader):.3f}')



def test(net,testloader,test_labels):
    criterion = nn.BCEWithLogitsLoss()
    loss = 0
    y_pred_list = []
    net.eval()
    print(len(testloader.dataset))
    with torch.no_grad():
        for item in testloader:
            X_batch, label = item
            X_batch = X_batch.to(device)
            y_test_pred = net(X_batch.unsqueeze(1))
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())
            loss += criterion(y_test_pred, label.unsqueeze(1).to(torch.float32)).item()
    print(len(y_pred_list))
    loss = loss/len(testloader)
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    print(classification_report(test_labels, y_pred_list))
    print("Loss: ",str(loss))
	

def main():
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, val_loader,testloader, test_labels = load_data()
    print("Start training")
    net=Net().to(device)
    train(net,trainloader, val_loader)
    print("Evaluate model")
    test(net, testloader,test_labels)


if __name__ == "__main__":
    main()
