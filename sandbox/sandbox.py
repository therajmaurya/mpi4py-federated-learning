from mpi4py import MPI
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# MPI init
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# synthetic dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# model
class SimpleLinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# model init
input_dim = X_train.shape[1]
local_model = SimpleLinearRegression(input_dim)

# optimizer and loss init
criterion = nn.MSELoss()
optimizer = optim.SGD(local_model.parameters(), lr=0.01)

# global model init
global_model = SimpleLinearRegression(input_dim)

# comm metrics
send_bytes = []
recv_bytes = []

# synchronize initial global model
for param, global_param in zip(local_model.parameters(), global_model.parameters()):
    data = param.data.numpy()
    comm.Bcast(data, root=0)
    send_bytes.append(len(data))
    global_param.data = torch.from_numpy(comm.bcast(param.data.numpy(), root=0))
    recv_bytes.append(len(param.data.numpy()))

# federated training using FedAvg
if rank == 0:
    print("Training using FedAvg...")
for epoch in range(10):
    # global model state
    for param, global_param in zip(local_model.parameters(), global_model.parameters()):
        param.data = global_param.data.clone()

    # local model
    for i in range(len(X_train_tensor)):
        optimizer.zero_grad()
        output = local_model(X_train_tensor[i])
        loss = criterion(output, y_train_tensor[i].view(-1, 1))
        loss.backward()
        optimizer.step()

    # average local models
    local_model_state = [param.data.clone() for param in local_model.parameters()]
    averaged_model_state = [torch.zeros_like(param) for param in local_model.parameters()]

    for param, global_param in zip(local_model_state, averaged_model_state):
        comm.Allreduce([param.numpy(), MPI.FLOAT], [global_param.numpy(), MPI.FLOAT], op=MPI.SUM)
        send_bytes.append(len(param.numpy()))
        recv_bytes.append(len(global_param.numpy()))

    for param in averaged_model_state:
        param /= size

    # update global model
    for global_param, param in zip(global_model.parameters(), averaged_model_state):
        global_param.data = param

    if rank == 0:
        print(f"Epoch {epoch + 1}, Global Model: {global_model.state_dict()}")

# federated training using FedSGD
if rank == 0:
    print("Training using FedSGD...")
for epoch in range(10):
    # load global model state
    for param, global_param in zip(local_model.parameters(), global_model.parameters()):
        param.data = global_param.data.clone()

    # train local model
    for i in range(len(X_train_tensor)):
        optimizer.zero_grad()
        output = local_model(X_train_tensor[i])
        loss = criterion(output, y_train_tensor[i].view(-1, 1))
        loss.backward()
        optimizer.step()

    # aggregate gradients
    local_gradients = [param.grad.clone() for param in local_model.parameters()]
    averaged_gradients = [torch.zeros_like(param.grad) for param in local_model.parameters()]

    for grad, global_grad in zip(local_gradients, averaged_gradients):
        comm.Allreduce([grad.numpy(), MPI.FLOAT], [global_grad.numpy(), MPI.FLOAT], op=MPI.SUM)
        send_bytes.append(len(grad.numpy()))
        recv_bytes.append(len(global_grad.numpy()))

    # update global model
    for global_param, grad in zip(global_model.parameters(), averaged_gradients):
        global_param.data -= 0.01 * grad / size

    if rank == 0:
        print(f"Epoch {epoch + 1}, Global Model: {global_model.state_dict()}")

# final global model
if rank == 0:
    print(f"Final global model: {global_model.state_dict()}")

# plot communication metrics
if rank == 0:
    plt.figure(figsize=(10, 5))
    plt.plot(send_bytes, label='Send Bytes', marker='o')
    plt.plot(recv_bytes, label='Receive Bytes', marker='x')
    plt.xlabel('Communication Step')
    plt.ylabel('Bytes')
    plt.title('Communication Metrics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
