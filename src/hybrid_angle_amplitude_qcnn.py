# Hybrid Quantum CNN using Angle + Amplitude Encoding
# Experiment: 5 Epoch Training
#
# Quantum Encoding:
# - Angle Embedding
# - Amplitude-inspired Feature Compression (8-dimensional latent vector)
#
# Quantum Circuit:
# - 8 Qubits
# - 6 Variational Layers
# - RY + RZ Rotations
# - Circular CNOT Entanglement
#
# Frameworks:
# - TensorFlow (Dataset Preparation)
# - PyTorch (Training)
# - PennyLane (Quantum Layer)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
import numpy as np

############################################
# Convert TensorFlow Dataset → NumPy
############################################
def tf_to_numpy(dataset):
    images = []
    labels = []
    for x, y in dataset:
        images.append(x.numpy())
        labels.append(y.numpy())
    images = np.concatenate(images)
    labels = np.concatenate(labels)
    return images, labels

X_train, y_train = tf_to_numpy(train_ds)
X_test, y_test   = tf_to_numpy(test_ds)

############################################
# Convert NumPy → PyTorch
############################################
X_train = torch.tensor(X_train).permute(0,3,1,2).float()
X_test  = torch.tensor(X_test).permute(0,3,1,2).float()
y_train = torch.tensor(y_train).long()
y_test  = torch.tensor(y_test).long()

############################################
# DataLoader
############################################
train_dataset = TensorDataset(X_train, y_train)
test_dataset  = TensorDataset(X_test, y_test)
trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
testloader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

############################################
# Quantum Circuit
############################################
n_qubits = 8
n_layers = 6
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    for layer in range(n_layers):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        for i in range(n_qubits):
            qml.RY(weights[layer][i][0], wires=i)
            qml.RZ(weights[layer][i][1], wires=i)
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[n_qubits-1, 0])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits, 2)}
quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

############################################
# Hybrid Quantum CNN
############################################
class HybridQCNN(nn.Module):
    def __init__(self):
        super(HybridQCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 8)
        self.quantum = quantum_layer
        self.fc3 = nn.Linear(8, 4)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.quantum(x)
        x = self.fc3(x)
        return x