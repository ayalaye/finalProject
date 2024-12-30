import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset,random_split
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from itertools import combinations, product
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.tensorboard import SummaryWriter
import datetime
from newDataSet import *
import matplotlib.pyplot as plt

   
class LightGlue(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_layers):
        super(LightGlue, self).__init__()

      
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(input_dim, 1)
       
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(1)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
       
        # Xavier initialization
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)
   

    def forward(self, xA, xB, posA, posB):

        # Normalization of input descriptors
        xA = (xA - xA.min()) / (xA.max() - xA.min())
        xB = (xB - xB.min()) / (xB.max() - xB.min())

        d_model = 128

        position_encoding_A = position_encoding_2d(posA, d_model)  
        position_encoding_B = position_encoding_2d(posB, d_model)

        position_encoding_A = torch.tensor(position_encoding_A, dtype=torch.float32, device=xA.device)        
        position_encoding_B = torch.tensor(position_encoding_B, dtype=torch.float32, device=xB.device)        


        xA = self.linear1(xA) + position_encoding_A
        xB = self.linear1(xB) + position_encoding_B
        
        xA=self.transformer_encoder(xA)
        xB=self.transformer_encoder(xB)
       
        # Add positional information to descriptors
        xA = self.linear1(xA)
        xB = self.linear1(xB)

        # Apply batch normalization
        xA = self.bn1(xA)
        xB = self.bn1(xB)

        # Compute similarity matrix
        S = torch.matmul(xA, xB.T)

        # Compute sigma values
        sigma_A = torch.sigmoid(self.linear2(xA))
        sigma_B = torch.sigmoid(self.linear2(xB))

        # Softmax over similarity matrix
        softmax_SkA = F.softmax(S, dim=1)
        softmax_SkB = F.softmax(S, dim=0)

        # Compute final similarity score matrix P
        P = sigma_A @ sigma_B.T * softmax_SkB * softmax_SkA
        P = (P - P.min()) / (P.max() - P.min())  # Normalize P to [0, 1]

        return sigma_A, sigma_B, P
   
   
# Generate 2D positional encodings for a set of 2D positions.
def position_encoding_2d(positions, d_model):
       
    assert d_model % 2 == 0
   
    position_enc = np.zeros((len(positions), d_model))
    d_model_half = d_model // 2  
   
    for idx, (x, y) in enumerate(positions):
        div_term = 10000 ** (2 * np.arange(d_model_half) / d_model_half)
        position_enc[idx, :d_model_half] = np.sin(x / div_term)
        position_enc[idx, d_model_half:] = np.cos(y / div_term)
   
    return position_enc


# Computes a custom loss for correspondence and unmatchable points.
def loss_(P, M, not_A, not_B, sigma_A, sigma_B):
    epsilon = 1e-8
    M = torch.tensor(M, dtype=torch.long)
    not_A = torch.tensor(not_A, dtype=torch.long)
    not_B = torch.tensor(not_B, dtype=torch.long)

    # Check shapes
    # print("Shape of P:", P.shape)
    # print("Shape of M:", M.shape)
    # print(sigma_B)

    if P.dim() == 1:
        correspondence_loss_sum = (torch.log(P[M] + epsilon).sum()) / len(M)
       
    else:
        correspondence_loss_sum = (torch.log(P[M[:, 0], M[:, 1]] + epsilon).sum()) / len(M)

    unmatchable_loss_A = 0.5 * (torch.log(1 - sigma_A[not_A] + epsilon)).sum() / len(not_A)
    unmatchable_loss_B = 0.5 * (torch.log(1 - sigma_B[not_B] + epsilon)).sum() / len(not_B)
   
    loss = correspondence_loss_sum + unmatchable_loss_A + unmatchable_loss_B
    return -loss


# Trains the model for one epoch and computes the average loss and accuracy (IoU).
def train(model, optimizer, loader,threshold):
    total_loss = 0
    total_iou = 0
    num_samples = 0

    for data in loader.dataset:
        if len(data['m']) == 0:
            # print("m=0")
            continue

        optimizer.zero_grad()  # Clear gradients.
        xA = torch.tensor(data['des1'], dtype=torch.float32)
        xB = torch.tensor(data['des2'], dtype=torch.float32)
        key1 = torch.tensor(data['key1'], dtype=torch.float32)
        key2 = torch.tensor(data['key2'], dtype=torch.float32)  

        sigma_A, sigma_B, P = model(xA, xB,key1,key2)
        loss = loss_(P, data['m'], data['notA'], data['notB'], sigma_A, sigma_B)

        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.

        total_loss += loss.item()

        M = torch.zeros_like(P)
        for match in data['m']:
            M[match[0], match[1]] = 1

        intersection = torch.sum((P > threshold) & (M == 1)).item()
        union = torch.sum((P > threshold) | (M == 1)).item()

        iou = intersection / union if union > 0 else 0
       
        total_iou += iou
        num_samples += 1

    avg_loss = total_loss / len(loader.dataset)
    avg_accuracy = total_iou / num_samples if num_samples > 0 else 0

    return avg_loss, avg_accuracy


# Evaluates the model on the test dataset and computes the average loss and accuracy (IoU).
def test(model, loader, threshold):
    model.eval()  
    total_loss = 0
    total_iou = 0
    num_samples = 0

    with torch.no_grad():
        for data in loader.dataset:
            if len(data['m']) == 0:
                continue
           
            xA = torch.tensor(data['des1'], dtype=torch.float32)
            xB = torch.tensor(data['des2'], dtype=torch.float32)

            key1 = torch.tensor(data['key1'], dtype=torch.float32)
            key2 = torch.tensor(data['key2'], dtype=torch.float32)

            sigma_A, sigma_B, P = model(xA, xB,key1,key2)

            M = torch.zeros_like(P)
            for match in data['m']:
                M[match[0], match[1]] = 1

            intersection = torch.sum((P > threshold) & (M == 1)).item()
            union = torch.sum((P > threshold) | (M == 1)).item()

            iou = intersection / union if union > 0 else 0
            print("accuracy ", iou)
            total_iou += iou
           
            loss = loss_(P, data['m'], data['notA'], data['notB'], sigma_A, sigma_B)
            print("loss ",loss)
            total_loss += loss.item()
           
            num_samples += 1
           
    avg_loss = total_loss / len(loader.dataset)
    avg_accuracy = total_iou / num_samples if num_samples > 0 else 0
    # print("acc:"+ avg_accuracy)
    return avg_loss, avg_accuracy  



# Example usage:

start=time.time()

dataset = MyDataset("")
num=10000
datasetName = "pets"

dataset=Subset(dataset, range(num))
dataset_size = len(dataset)

train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

num_workers=16
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=num_workers)#numOfWorkers
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,num_workers=num_workers)

input_dim = 128  
output_dim = 128  
nhead=2
num_layers=2
model = LightGlue(input_dim, output_dim,nhead,num_layers)

lr=0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

# thresholds=[0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]

threshold=0.25
epochs = []
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(0, 40):
    train_loss, train_accuracy = train(model, optimizer, train_loader,threshold)
    test_loss, test_accuracy = test(model, test_loader, threshold)

    epochs.append(epoch)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    print("epoch: {} train_loss: {:.4f}, test_loss: {:.4f}, train_accuracy: {:.4f}, test_accuracy: {:.4f}".format(
        epoch, train_loss, test_loss, train_accuracy, test_accuracy
    ))
  
end=time.time()

# Set up a single figure with two subplots side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot the loss graph on the first subplot
ax1.plot(epochs, train_losses, marker='o', color='b', label='Train Loss')
ax1.plot(epochs, test_losses, marker='o', color='r', label='Test Loss')
ax1.set_title(f'Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_xticks(epochs)
ax1.grid()
ax1.legend()

# Plot the accuracy graph on the second subplot
ax2.plot(epochs, train_accuracies, marker='o', color='b', label='Train Accuracy')
ax2.plot(epochs, test_accuracies, marker='o', color='r', label='Test Accuracy')
ax2.set_title(f'Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_xticks(epochs)
ax2.grid()
ax2.legend()

# calculate time
delta = end - start  
hours = delta / 3600  
hours_rounded = round(hours, 3)

# Add a main title for the entire figure
plt.suptitle(f'Data: {datasetName} | Amount: {num} | Threshold: {threshold} | LearningRate: {lr} | nhead: {nhead} | Layers: {num_layers} | numWorkers: {num_workers} | Time: {hours_rounded}' , fontsize=16)

# Get the current timestamp
current_time = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

# Save the combined figure to the specified directory
fig.savefig(f'./petsGraphs/loss_accuracy_{current_time}.png')

# Close the plot to free up memory
plt.close(fig)
print("Time: ",hours_rounded)

# Save the model with a specified path
model_path = f'./petsGraphs/model_{current_time}.pt'
torch.save(model.state_dict(), model_path)



# print("-------------------------------------------testing-------------------------------------")
# test_dataset=Subset(test_dataset, range(30))
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=num_workers)
# test_loss, test_accuracy = test(model, test_loader, threshold)
# print("avg_loss ",test_loss)
# print("avg acc ",test_accuracy)

# print("----------------------------------------------one picture----------------------")
# with open('data/data_Abyssinian_9.json', 'r') as file:
#     data = json.load(file)
#     model.eval()  

#     with torch.no_grad():
           
#         xA = torch.tensor(data['des1'], dtype=torch.float32)
#         xB = torch.tensor(data['des2'], dtype=torch.float32)

#         key1 = torch.tensor(data['key1'], dtype=torch.float32)
#         key2 = torch.tensor(data['key2'], dtype=torch.float32)

#         sigma_A, sigma_B, P = model(xA, xB,key1,key2)

#         M = torch.zeros_like(P)
#         for match in data['m']:
#             M[match[0], match[1]] = 1

#         intersection = torch.sum((P > threshold) & (M == 1)).item()
#         union = torch.sum((P > threshold) | (M == 1)).item()

#         iou = intersection / union if union > 0 else 0
#         print("accuracy ", iou)
        
#         loss = loss_(P, data['m'], data['notA'], data['notB'], sigma_A, sigma_B)
#         print("loss ",loss)
     


