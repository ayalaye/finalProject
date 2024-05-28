import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import tensorflow as tf
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
#
############## TENSORBOARD ########################
import sys
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('logs/mnist1')
###################################################


# Create a Model Class that inherits nn.Module
class Model(nn.Module):
  # Input layer (image) -->
  # Hidden Layer1 (number of neurons) -->
  # H2 (n) -->
  # H3
  # output (vector of probabilities )
  def __init__(self, in_features=784, h1=256, h2=128, out_features=10):
    super().__init__() # instantiate our nn.Module
    self.fc1 = nn.Linear(in_features, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)
    x=F.log_softmax(x,dim=1)
    return x

# Pick a manual seed for randomization
torch.manual_seed(41)
# Create an instance of model
model = Model()

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ]))
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ]))

# Set batch size for training
batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False)


examples = iter(test_loader)
example_data, example_targets = next(examples)

# Set the criterion of model to measure the error, how far off the predictions are from the data
criterion = nn.CrossEntropyLoss()
# Choose Adam Optimizer, lr = learning rate (if error doesn't go down after a bunch of iterations (epochs), lower our learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# Assuming your model is named 'model', add its computational graph to TensorBoard
# writer.add_graph(model, example_data.reshape(-1, 28*28))

# Train our model!
print("Train our model")
epochs = 1
losses = []
for epoch in range(epochs):
  running_loss = 0.0
  for i, (images, labels) in enumerate(train_loader):
    images = images.view(-1, 28 * 28)  # Flatten the images
    y_pred = model.forward(images) # Get predicted results

  # Measure the loss/error, gonna be high at first
    loss = criterion(y_pred, labels) # predicted values vs the y_train

  # Keep Track of our losses
    writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + i)

 # Calculate running loss
    running_loss += loss.item()

  # print every 10 epoch
    if (i+1) % 100 == 0:
      print(f'Epoch: {epoch+1} batch: {i+1} and loss: {loss}')
      # Write loss to TensorBoard
      # Write loss to TensorBoard
      running_loss = 0.0

  # Do some back propagation: take the error rate of forward propagation and feed it back
  # thru the network to fine tune the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # losses.append(loss.detach().numpy())
  losses.append(loss.detach().numpy())


# test our model
print("Test our model")

correct = 0
with torch.no_grad():
  #for i, data in enumerate(flatten_test_images):
  for i, (images, labels) in enumerate(test_loader):
    images = images.view(-1, 28 * 28)  # Flatten the images
    y_val = model.forward(images)

    # print(f'{i+1}.)   \t {labels.item()}  {y_val.argmax().item()}')

    # Correct or not
    if y_val.argmax().item() == labels:
      correct +=1
sucsess=correct/(len(test_loader))
print(f'We got {sucsess}% correct!')

plt.plot(range(len(losses)), losses)
plt.ylabel("loss/error")
plt.xlabel('Epoch')
writer.close()
