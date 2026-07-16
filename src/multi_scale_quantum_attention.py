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
quantum_layer_low  = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
quantum_layer_high = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

############################################
# Multi-Scale Quantum Attention QCNN
############################################
class MSQAHybridQCNN(nn.Module):
    def __init__(self):
        super(MSQAHybridQCNN, self).__init__()

        # Low scale block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4,4))
        )

        # High scale block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4,4))
        )

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)

        # Reduce features for quantum circuits
        self.fc_low  = nn.Linear(32 * 4 * 4, 8)
        self.fc_high = nn.Linear(64 * 4 * 4, 8)

        # Quantum attention circuits
        self.quantum_low  = quantum_layer_low
        self.quantum_high = quantum_layer_high

        # Final classifier → 4 classes
        self.fc_final = nn.Linear(16, 4)

    def forward(self, x):
        # Multi-scale features
        low_features  = self.conv1(x)
        high_features = self.conv2(low_features)

        # Flatten
        low_features  = self.dropout(self.flatten(low_features))
        high_features = self.dropout(self.flatten(high_features))

        # Reduce dimension
        low_features  = torch.tanh(self.fc_low(low_features))
        high_features = torch.tanh(self.fc_high(high_features))

        # Quantum attention
        att_low  = torch.sigmoid(self.quantum_low(low_features))
        att_high = torch.sigmoid(self.quantum_high(high_features))

        # Apply attention
        low_features  = low_features  + low_features  * att_low
        high_features = high_features + high_features * att_high

        # Fusion
        fused = torch.cat((low_features, high_features), dim=1)
        out = self.fc_final(fused)
        return out

model = MSQAHybridQCNN()