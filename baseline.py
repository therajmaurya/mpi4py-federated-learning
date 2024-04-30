import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

# Loading the data
data = np.loadtxt("training_data.csv", delimiter=",", skiprows=1)
x = data[:, 0]
y = data[:, 1]

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
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# plotting the loss curve
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig("images/baseline/Loss Curve.png")

# Evaluating the model
y_pred = model(x_tensor)
final_loss = loss_fn(y_pred, y_tensor)
r2 = r2_score(y, y_pred.detach().numpy())
mae = mean_absolute_error(y, y_pred.detach().numpy())

print(f'Final loss: {final_loss.item()}')
print(f'R-squared: {r2}')
print(f'Mean Absolute Error: {mae}')

# plotting final datapoints result
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Original data')
plt.scatter(x, y_pred.detach().numpy(), label='Fitted line')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Data vs Fitted Line')
plt.savefig("images/baseline/Model Predictions Curve.png")
