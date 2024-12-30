# main.py
import data_loader
import torch

data_path = './resize_photos'

# Load the dataset once
dataset = data_loader.MyDataset(data_path)

# Access samples without reloading the dataset
for i in range(10):
    sample = dataset[i]
    # Use the sample data here
    # ...

# Use the dataset object for training or other tasks
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
# for batch in dataloader:
#     # Process the batch of data here
    
