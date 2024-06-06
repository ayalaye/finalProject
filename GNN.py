

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
        print(S[:5,:5])
        sigma_A = torch.sigmoid(self.linear2(xA))
        sigma_B = torch.sigmoid(self.linear2(xB))
        
        softmax_SkA = F.softmax(S/100000, dim=1)
        softmax_SkB = F.softmax(S/100000, dim=0)
        P = torch.zeros_like(S)
        print(sigma_A[:10])
        print(sigma_B[:10]) 
        print(softmax_SkA)
        print(softmax_SkB)

        P = sigma_A @ sigma_B.T * softmax_SkB * softmax_SkA


        return sigma_A, sigma_B, P
    
def loss(P, M, not_A, not_B, sigma_A, sigma_B):
    epsilon = 1e-8
    loss=0

    M = torch.tensor(M, dtype=torch.long)
    not_A = torch.tensor(not_A, dtype=torch.long)
    not_B = torch.tensor(not_B, dtype=torch.long)
    print("p:",P)
    # print("sigma_a",sigma_A)
    correspondence_loss_sum = ((torch.log(P[M[:, 0], M[:, 1]]+epsilon) ).sum())/len(M)
    unmatchable_loss_A = 0.5 * ((torch.log(1 - sigma_A[not_A]+epsilon)).sum()) / len(not_A)
    unmatchable_loss_B = 0.5 * ((torch.log(1 - sigma_B[not_B]+epsilon)).sum()) / len(not_B)
    loss = correspondence_loss_sum + unmatchable_loss_A + unmatchable_loss_B 

    return -loss
# def loss(P, M, not_A, not_B, sigma_A, sigma_B):
#     loss = 0
    
#     # Correspondences loss term
#     M_indices = torch.tensor(M, dtype=torch.long)
#     correspondence_loss = torch.log(P)

#     correspondence_loss_sum = correspondence_loss[M_indices[:, 0], M_indices[:, 1]].sum() / len(M_indices)
    
#     unmatchable_loss_A = 0.5 * torch.log(1 - sigma_A[not_A]).sum() / len(not_A)
#     unmatchable_loss_B = 0.5 * torch.log(1 - sigma_B[not_B]).sum() / len(not_B)
    
#     loss = correspondence_loss_sum + unmatchable_loss_A + unmatchable_loss_B 

#     return -loss

# def loss(P, M, not_A, not_B, sigma_A, sigma_B):
#     loss = 0

#     # Correspondences loss term (assuming M is a LongTensor of indices)
#     M_indices = torch.tensor(M, dtype=torch.long)  # Convert M to LongTensor
#     correspondence_loss = -F.log_softmax(P,dim=1)  # Negative log-softmax for cross-entropy
#     correspondence_loss_sum = correspondence_loss[M_indices[:, 0], M_indices[:, 1]].sum() / len(M_indices)
#     # Unmatchable loss terms
#     unmatchable_loss_A = -torch.mean(F.log_softmax(torch.cat((sigma_A, 1 - sigma_A), dim=1), dim=1)[not_A])  # Explicit dim=1
#     unmatchable_loss_B = -torch.mean(F.log_softmax(torch.cat((sigma_B, 1 - sigma_B), dim=1), dim=1)[not_B])  # Explicit dim=1

#     loss = correspondence_loss_sum + unmatchable_loss_A + unmatchable_loss_B

#     return loss
    

        


# Example usage:

data_path = './resize_photos'
   
dataset = MyDataset(data_path)


input_dim = 128  # Example input dimension// ?????
output_dim = 128  # Example output dimension// ?????


model = LightGlue(input_dim, output_dim)
# xA = torch.randn(10, input_dim)  # Example input tensor A
# xB = torch.randn(10, input_dim)  # Example input tensor B
xA=torch.tensor(dataset[0]['des1'],dtype=torch.float32)
xB=torch.tensor(dataset[0]['des2'],dtype=torch.float32)
print(xA[0,0:5])
sigma_A, sigma_B, P = model(xA, xB)
loss = loss(P, dataset[0]['m'], dataset[0]['notA'], dataset[0]['notB'], sigma_A, sigma_B)
print(loss)











# class GAT(torch.nn.Module):
#     def __init__(self, in_channels=128, out_channels=128):
#         super(GAT, self).__init__()
#         self.DB_percentage = torch.nn.Parameter(torch.ones(1) * 0.4, requires_grad=True)
#         self.hid = 1
#         self.in_head = 128
#         self.out_head = 1

#         self.conv1 = GATConv(in_channels, self.hid, heads=self.in_head, dropout=0.6)
#         self.conv2 = GATConv(self.hid * self.in_head, out_channels, concat=False, heads=self.out_head, dropout=0.6)
        

# def forward(self, data):
#     iters = 1
#     des1, des2 = data['des1'], data['des2']
#     # des1 = F.normalize(des1, dim = 0)
#     # des2 = F.normalize(des2, dim = 0)

#     # inside_edge, cross_edge = self.get_edge_index(desc1, desc2)

#     x = torch.Tensor(np.concatenate((des1, des2)))
#     # for i in range(iters):
#     #     print('x shape: ', x.shape)
#     #     # print("x before conv1: ", x)
#     #     x = self.conv1(x, inside_edge)
#     #     # print("x after conv1: ", x)
#     #     print('x shape: ', x.shape)
#     #     x = F.elu(x)
#     # x = self.conv2(x, cross_edge)

#     xA = x[0:len(des1)]
#     xB = x[len(des1):]

#     S = torch.matmul(self.linear1(xA), self.linear1(xB).transpose(0, 1))

#     sigma_A = torch.sigmoid(self.linear2(xA))
#     sigma_B = torch.sigmoid(self.linear2(xB))

#     softmax_SkA = F.softmax(S, dim=1)
#     softmax_SkB = F.softmax(S, dim=0)

#     P = torch.zeros_like(S)
#     for i in range(S.size(0)):
#         for j in range(S.size(1)):
#             P[i, j] = sigma_A[i] * sigma_B[j] * softmax_SkB[i, :].sum() * softmax_SkA[:, j].sum()

#     return P
#     # p_match, match = sinkhorn_match2(des1, des2, self.DB_percentage)
#     # return p_match, match