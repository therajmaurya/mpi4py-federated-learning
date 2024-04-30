from mpi4py import MPI
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

# MPI init
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Loading data
data = np.loadtxt("training_data.csv", delimiter=",", skiprows=1)
x = data[:, 0]
y = data[:, 1]

# Each core reads all the rows where row%size = rank
indices = np.where(np.arange(len(x)) % size == rank)[0]
x = x[indices]
y = y[indices]

# Converting the data to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)


class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()
        self.linear1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

model = SimpleLinearRegression()

# Define your loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-6)

# Create DataLoader for efficient batch processing
dataset = TensorDataset(x_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train your model
losses = []

for epoch in range(500):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    if epoch % 100 == 99:
        print(f'Rank {rank}, Epoch {epoch+1}, Loss: {loss.item():.4f}')

    # After each epoch, we average the models using FedAvg
    for param in model.parameters():
        data = param.data.numpy() if rank == 0 else None
        data = comm.bcast(data, root=0)
        param.data = torch.tensor(data)

# Save the plot of original data and prediction on the model
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Original data')
y_pred = model(x_tensor)
plt.scatter(x, y_pred.detach().numpy(), label='Fitted line')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Data vs Fitted Line')
plt.savefig(f'plot_rank_{rank}.png')
