

import numpy as np
import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from itertools import combinations, product
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.tensorboard import SummaryWriter
from newDataSet import * 
   

class LightGlue(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LightGlue, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        init.xavier_uniform_(self.linear1.weight)
        self.linear2 = nn.Linear(input_dim, 1)
        init.xavier_uniform_(self.linear2.weight)

    def forward(self, xA, xB):
        
        S = torch.matmul(self.linear1(xA), self.linear1(xB).T)
        # print(S[:5,:5])
        sigma_A = torch.sigmoid(self.linear2(xA))
        sigma_B = torch.sigmoid(self.linear2(xB))
        
        softmax_SkA = F.softmax(S/100000, dim=1)
        softmax_SkB = F.softmax(S/100000, dim=0)
        P = torch.zeros_like(S)
        # print(sigma_A[:10])
        # print(sigma_B[:10]) 
        # print(softmax_SkA)
        # print(softmax_SkB)

        P = sigma_A @ sigma_B.T * softmax_SkB * softmax_SkA


        return sigma_A, sigma_B, P
    
def loss_(P, M, not_A, not_B, sigma_A, sigma_B):
    epsilon = 1e-8
    loss=0

    M = torch.tensor(M, dtype=torch.long)
    not_A = torch.tensor(not_A, dtype=torch.long)
    not_B = torch.tensor(not_B, dtype=torch.long)
    # print("p:",P)
    # print("sigma_a",sigma_A)
    correspondence_loss_sum = ((torch.log(P[M[:, 0], M[:, 1]]+epsilon) ).sum())/len(M)
    unmatchable_loss_A = 0.5 * ((torch.log(1 - sigma_A[not_A]+epsilon)).sum()) / len(not_A)
    unmatchable_loss_B = 0.5 * ((torch.log(1 - sigma_B[not_B]+epsilon)).sum()) / len(not_B)
    loss = correspondence_loss_sum + unmatchable_loss_A + unmatchable_loss_B 

    return -loss
def train(model, optimizer, loader):
    total_loss = 0
    for data in loader.dataset:
        optimizer.zero_grad()  # Clear gradients.
        xA=torch.tensor(data['des1'],dtype=torch.float32)
        xB=torch.tensor(data['des2'],dtype=torch.float32)
        sigma_A, sigma_B, P = model(xA, xB)

        loss = loss_(P, data['m'], data['notA'], data['notB'], sigma_A, sigma_B)

        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        
        total_loss += loss.item()

    return total_loss / len(loader.dataset)       


# Example usage:



writer = SummaryWriter('logs/log6')

data_path = './resize_photos'
   
dataset = MyDataset(data_path)




train_dataset=dataset
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)

input_dim = 128  # Example input dimension// ?????
output_dim = 128  # Example output dimension// ?????


model = LightGlue(input_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(1, 2):
    loss = train(model, optimizer, train_loader)
    print (loss)
    writer.add_scalar('Training Loss', loss, epoch)

writer.close()

# xA=torch.tensor(dataset[0]['des1'],dtype=torch.float32)
# xB=torch.tensor(dataset[0]['des2'],dtype=torch.float32)
# print(xA[0,0:5])
# sigma_A, sigma_B, P = model(xA, xB)
# loss = loss(P, dataset[0]['m'], dataset[0]['notA'], dataset[0]['notB'], sigma_A, sigma_B)
# print(loss)







# class GAT(torch.nn.Module):
#     def __init__(self, in_channels=128, out_channels=128):
#         super(GAT, self).__init__()
#         self.DB_percentage = torch.nn.Parameter(torch.ones(1) * 0.4, requires_grad=True)
#         self.hid = 1
#         self.in_head = 128
#         self.out_head = 1

#         self.conv1 = GATConv(in_channels, self.hid, heads=self.in_head, dropout=0.6)
#         self.conv2 = GATConv(self.hid * self.in_head, out_channels, concat=False, heads=self.out_head, dropout=0.6)
        

