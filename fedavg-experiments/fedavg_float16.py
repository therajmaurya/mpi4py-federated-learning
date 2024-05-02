# This code will work only on the system which supports half precision computation
# If we run on a CPU, it will give "RuntimeError: "mse_backward_cpu_out" not implemented for 'Half'",
# as half precision is not supported on normal CPUs

from mpi4py import MPI
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

# setting seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# MPI init
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Loading data
data = np.loadtxt("training_data.csv", delimiter=",", skiprows=1)
x = data[:, 0]
y = data[:, 1]

# each core reads all the rows where row%size = rank
indices = np.where(np.arange(len(x)) % size == rank)[0]
x = x[indices]
y = y[indices]

x_tensor = torch.tensor(x, dtype=torch.float16).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.float16).unsqueeze(1)

# Load entire data as test data
test_data = np.loadtxt("training_data.csv", delimiter=",", skiprows=1)
x_test = data[:, 0]
y_test = data[:, 1]

x_test_tensor = torch.tensor(x_test, dtype=torch.float16).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float16).unsqueeze(1)

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

def count_parameters_and_type(model):
    count_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    d_type = next(model.parameters()).dtype
    d_size = 0
    if str(d_type) == "torch.float64" or str(d_type) == "torch.int64":
        d_size = 8
    elif str(d_type) == "torch.float32" or str(d_type) == "torch.int32":
        d_size = 4
    elif str(d_type) == "torch.float16" or str(d_type) == "torch.int16":
        d_size = 2
    elif str(d_type) == "torch.uint8":
        d_size = 1
    else:
        d_size = 2
    return count_params, d_size, d_type

model = SimpleLinearRegression()
# converting the model to half precision
model.half()

num_params, d_size, d_type = count_parameters_and_type(model)
print(f'The model has {num_params} parameters each having type {d_type} with size {d_size} bytes.')

# loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-6)

# DataLoader for efficient batch processing
dataset = TensorDataset(x_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# model training
losses = []
message_counts = []
message_sizes = []
cumulative_message_count = 0
cumulative_message_size = 0
for epoch in range(1000):
    for inputs, targets in dataloader:
        # coonverting inputs and targets to half precision
        inputs = inputs.half()
        targets = targets.half()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    if epoch % 100 == 99:
        print(f'Rank {rank}, Epoch {epoch+1}, Loss: {loss.item():.4f}')

# model eval
# converting test tensors to half precision
x_test_tensor = x_test_tensor.half()
y_test_tensor = y_test_tensor.half()

y_test_pred = model(x_test_tensor)
final_loss = loss_fn(y_test_pred, y_test_tensor)

# converting predictions back to float32 for evaluation, model ran in float16 precision only
y_test_pred = y_test_pred.float()
r2 = r2_score(y_test, y_test_pred.detach().numpy())
mae = mean_absolute_error(y_test, y_test_pred.detach().numpy())

print(f'Rank {rank}, Final loss: {final_loss.item()}')
print(f'Rank {rank}, R-squared: {r2}')
print(f'Rank {rank}, Mean Absolute Error: {mae}')

# plotting
plt.figure(figsize=(10, 6))
plt.scatter(x_test, y_test, label='Original data')
plt.scatter(x_test, y_test_pred.detach().numpy(), label='Fitted line')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Data vs Fitted Line')
plt.savefig(f'images/fedavg/Model Predictions Curve Rank {rank} Before Reducing.png')

# after training, averaging the models using FedAvg
avg_params = []
for param in model.parameters():
    data = param.data.numpy()
    avg_data = np.empty_like(data)
    comm.Allreduce(data, avg_data, op=MPI.SUM)
    avg_data /= size
    avg_params.append(avg_data)

# since model synchronization is happening every iteration, we need to count the parameters every iteraation
num_params, d_size, _ = count_parameters_and_type(model)
cumulative_message_count += num_params
# We have implement this NN in torch.float16 precision since params needs lesser precision than gradients
cumulative_message_size += d_size * num_params
message_counts.append(cumulative_message_count)
message_sizes.append(cumulative_message_size)

# new model with the averaged parameters
model = SimpleLinearRegression()
for param, avg_param in zip(model.parameters(), avg_params):
    param.data = torch.tensor(avg_param)

# model eval after reducing
y_test_pred = model(x_test_tensor)
final_loss = loss_fn(y_test_pred, y_test_tensor)
r2 = r2_score(y_test, y_test_pred.detach().numpy())
mae = mean_absolute_error(y_test, y_test_pred.detach().numpy())

print(f'After Reducing at Rank {rank}, Final loss: {final_loss.item()}')
print(f'After Reducing at Rank {rank}, R-squared: {r2}')
print(f'After Reducing at Rank {rank}, Mean Absolute Error: {mae}')

# plotting
plt.figure(figsize=(10, 6))
plt.scatter(x_test, y_test, label='Original data')
plt.scatter(x_test, y_test_pred.detach().numpy(), label='Fitted line')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Data vs Fitted Line')
plt.savefig(f'images/fedavg/Model Predictions Curve Rank {rank} After Reducing.png')

# plotting the loss curve
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig("images/fedavg/Loss Curve.png")

# plotting the message size
plt.figure(figsize=(10, 6))
plt.scatter(range(1, len(message_sizes)+1), message_sizes)
plt.title('Cumulative Message Size Over Time')
plt.xlabel('Epoch')
plt.ylabel('Cumulative Message Size (bytes)')
plt.savefig(f"images/fedavg/Message Size Over Time at Rank {rank}.png")

# plotting the message complexity i.e. number of messages exchanged
plt.figure(figsize=(10, 6))
plt.scatter(range(1, len(message_counts)+1), message_counts)
plt.title('Cumulative Message Complexity Over Time')
plt.xlabel('Epoch')
plt.ylabel('Cumulative Number of Messages')
plt.savefig(f"images/fedavg/Message Complexity Over Time at Rank {rank}.png")