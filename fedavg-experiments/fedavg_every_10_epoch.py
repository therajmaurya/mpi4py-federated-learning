from mpi4py import MPI
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

# seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

#  MPI init
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

# test data loading (trating entire data as test data for now)
test_data = np.loadtxt("training_data.csv", delimiter=",", skiprows=1)
x_test = data[:, 0]
y_test = data[:, 1]

x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

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
for epoch in range(1000):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
    losses.append(loss.item())

    # Every 10 epochs, we average the models using FedAvg
    if epoch % 10 == 9:
        if epoch % 100 == 99:  # every 100 epochs we are tracking metrics
            print(f'Rank {rank}, Epoch {epoch+1}, Loss: {loss.item():.4f}')

            # model eval locally
            y_test_pred = model(x_test_tensor)
            final_loss = loss_fn(y_test_pred, y_test_tensor)
            r2 = r2_score(y_test, y_test_pred.detach().numpy())
            mae = mean_absolute_error(y_test, y_test_pred.detach().numpy())

            print(f'Before reducing, Rank {rank} at epoch {epoch+1}, Final loss: {final_loss.item()}')
            print(f'Before reducing, Rank {rank} at epoch {epoch+1}, R-squared: {r2}')
            print(f'Before reducing, Rank {rank} at epoch {epoch+1}, Mean Absolute Error: {mae}')

            # plotting
            plt.figure(figsize=(10, 6))
            plt.scatter(x_test, y_test, label='Original data')
            plt.scatter(x_test, y_test_pred.detach().numpy(), label='Fitted line')
            plt.legend()
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Original Data vs Fitted Line')
            plt.savefig(f'plot_rank_{rank}_epoch_{epoch+1}_before_reducing.png')

        # FedAvg
        avg_params = []
        for param in model.parameters():
            data = param.data.numpy()
            avg_data = np.empty_like(data)
            comm.Allreduce(data, avg_data, op=MPI.SUM)
            avg_data /= size
            avg_params.append(avg_data)

        # new model with the averaged parameters
        model = SimpleLinearRegression()
        for param, avg_param in zip(model.parameters(), avg_params):
            param.data = torch.tensor(avg_param)

        if epoch % 100 == 99:
            # model eval after reducing
            y_test_pred = model(x_test_tensor)
            final_loss = loss_fn(y_test_pred, y_test_tensor)
            r2 = r2_score(y_test, y_test_pred.detach().numpy())
            mae = mean_absolute_error(y_test, y_test_pred.detach().numpy())

            print(f'After reducing, Rank {rank} at epoch {epoch+1}, Final loss: {final_loss.item()}')
            print(f'After reducing, Rank {rank} at epoch {epoch+1}, R-squared: {r2}')
            print(f'After reducing, Rank {rank} at epoch {epoch+1}, Mean Absolute Error: {mae}')

            # plotting
            plt.figure(figsize=(10, 6))
            plt.scatter(x_test, y_test, label='Original data')
            plt.scatter(x_test, y_test_pred.detach().numpy(), label='Fitted line')
            plt.legend()
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Original Data vs Fitted Line')
            plt.savefig(f'plot_rank_{rank}_epoch_{epoch+1}_after_reducing.png')
