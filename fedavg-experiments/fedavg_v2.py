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

# data loading
data = np.loadtxt("training_data.csv", delimiter=",", skiprows=1)
x = data[:, 0]
y = data[:, 1]

# each core reads all the rows where row%size = rank
indices = np.where(np.arange(len(x)) % size == rank)[0]
x = x[indices]
y = y[indices]

x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# model
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

# loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-6)

# DataLoader for efficient batch processing
dataset = TensorDataset(x_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# model training
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

# after training, we average the models using FedAvg
for param in model.parameters():
    data = param.data.numpy() if rank == 0 else None
    data = comm.bcast(data, root=0)
    param.data = torch.tensor(data)

# model eval locally
y_pred = model(x_tensor)
final_loss = loss_fn(y_pred, y_tensor)
r2 = r2_score(y, y_pred.detach().numpy())
mae = mean_absolute_error(y, y_pred.detach().numpy())

print(f'Rank {rank}, Final loss: {final_loss.item()}')
print(f'Rank {rank}, R-squared: {r2}')
print(f'Rank {rank}, Mean Absolute Error: {mae}')

# plotting
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Original data')
plt.scatter(x, y_pred.detach().numpy(), label='Fitted line')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Data vs Fitted Line')
plt.savefig(f'plot_rank_{rank}_local.png')

# reduce the models using FedAvg
for param in model.parameters():
    data = param.data.numpy()
    data = comm.reduce(data, op=MPI.SUM, root=0)
    if rank == 0:
        param.data = torch.tensor(data / size)

# model eval after reducing
if rank == 0:
    y_pred = model(x_tensor)
    final_loss = loss_fn(y_pred, y_tensor)
    r2 = r2_score(y, y_pred.detach().numpy())
    mae = mean_absolute_error(y, y_pred.detach().numpy())

    print(f'After reducing, Final loss: {final_loss.item()}')
    print(f'After reducing, R-squared: {r2}')
    print(f'After reducing, Mean Absolute Error: {mae}')

    # plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Original data')
    plt.scatter(x, y_pred.detach().numpy(), label='Fitted line')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Original Data vs Fitted Line')
    plt.savefig('plot_after_reducing.png')
