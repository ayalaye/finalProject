
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
from newDataSet import * 
   
# class LightGlue:
#     def __init__(self, input_dim, output_dim):
#         # Initialize LightGlue with input and output dimensions
#         self.input_dim = input_dim
#         self.output_dim = output_dim
        
#     def compute_correspondence_loss(self, P, M, A_unmatchable, B_unmatchable, sigma_A, sigma_B):
#         L = P.shape[0]  # Number of layers
#         loss = 0
        
#         # Compute correspondence loss term
#         correspondence_loss = -np.log(P)
#         correspondence_loss_sum = np.sum(correspondence_loss[:, M[:, 0], M[:, 1]])
#         unmatchable_loss_A = 0.5 * np.sum(np.log(1 - P[:, A_unmatchable])) / len(A_unmatchable)
#         unmatchable_loss_B = 0.5 * np.sum(np.log(1 - P[:, :, B_unmatchable])) / len(B_unmatchable)
        
#         loss += correspondence_loss_sum + unmatchable_loss_A + unmatchable_loss_B
        
#         return loss / L
    
#     def train_confidence_classifier(self, predictions, final_predictions):
#         L = len(predictions)  # Number of layers
#         loss = 0
        
#         # Compute binary cross-entropy loss for each layer
#         binary_cross_entropy = -(final_predictions * np.log(predictions) + (1 - final_predictions) * np.log(1 - predictions))
#         loss = np.mean(binary_cross_entropy[:-1])  # Exclude the last layer
        
#         return loss / (L - 1)


class LightweightHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LightweightHead, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(input_dim, 1)

    def forward(self, xA, xB):
        S = torch.matmul(self.linear1(xA), self.linear1(xB).transpose(0, 1))

        sigma_A = torch.sigmoid(self.linear2(xA))
        sigma_B = torch.sigmoid(self.linear2(xB))

        softmax_SkA = F.softmax(S, dim=1)
        softmax_SkB = F.softmax(S, dim=0)

        P = torch.zeros_like(S) 

        P = sigma_A * sigma_B * softmax_SkB * softmax_SkA
        return sigma_A, sigma_B, P
    

def loss(P, M, not_A, not_B, sigma_A, sigma_B):
    loss = 0
    
    # Correspondences loss term
    M_indices = torch.tensor(M)  # Convert M to tensor
    correspondence_loss = torch.log(P)
    correspondence_loss_sum = torch.sum(correspondence_loss[:, M_indices[:, 0], M_indices[:, 1]]) / len(M_indices)
    
    # Unmatchable points loss terms
    unmatchable_loss_A = 0.5 * torch.sum(torch.log(1 - sigma_A[:, not_A])) / len(not_A) 
    unmatchable_loss_B = 0.5 * torch.sum(torch.log(1 - sigma_B[:, not_B])) / len(not_B) 
    
    loss = correspondence_loss_sum + unmatchable_loss_A + unmatchable_loss_B 

    
    return -loss
    

        


# Example usage:
input_dim = 128  # Example input dimension// ?????
output_dim = 64  # Example output dimension// ?????


model = LightweightHead(input_dim, output_dim)
xA = torch.randn(10, input_dim)  # Example input tensor A
xB = torch.randn(10, input_dim)  # Example input tensor B
sigma_A, sigma_B, P = model(xA, xB)
# loss = loss(P, m, not_A, not_B, sigma_A, sigma_B)
print(P)





# count=0
# data_path = './resize_photos'
# dic=[]
# # Iterate through folders in the dataset
# for i in range(100):
#     images, matrix = load_images_and_matrix(data_path)
#     dic.append((images,matrix))

# # Create an instance of the custom dataset class
# dataset = MyDataset(dic)
# # Example usage:
# # Accessing the first sample in the dataset
# for i in range(100):
#     print("dicti:", dataset[i])

# # Get the length of the dataset
# dataset_length = len(dataset)
# print("Length of the dataset:", dataset_length)








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