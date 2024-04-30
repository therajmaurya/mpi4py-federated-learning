import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
plt.ioff()

def load_data(file_path):
    data = genfromtxt(file_path, delimiter=',', skip_header=1)
    print(data)
    print("Shape of loaded data:", data.shape)  # Add this line for troubleshooting
    X = data[:, :-1].astype(np.float16)
    y = data[:, -1].astype(np.float16)
    return torch.tensor(X), torch.tensor(y)

# Load data
X_train, y_train = load_data("training_data.csv")
print(X_train)