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
# Quantum Device
############################################
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

############################################
# Quantum Circuit 1 : Angle Encoding
############################################
@qml.qnode(dev, interface="torch")
def quantum_circuit_angle(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
    for l in range(6):
        for i in range(n_qubits):
            qml.RY(weights[l, i, 0], wires=i)
            qml.RZ(weights[l, i, 1], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

############################################
# Quantum Circuit 2 : Amplitude Encoding
############################################
@qml.qnode(dev, interface="torch")
def quantum_circuit_amp(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
    for l in range(6):
        for i in range(n_qubits):
            qml.RY(weights[l, i, 0], wires=i)
            qml.RZ(weights[l, i, 1], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

############################################
# Convert Circuits to Torch Layers
############################################
weight_shapes = {"weights": (6, n_qubits, 2)}
quantum_layer_angle = qml.qnn.TorchLayer(quantum_circuit_angle, weight_shapes)
quantum_layer_amp   = qml.qnn.TorchLayer(quantum_circuit_amp,   weight_shapes)

############################################
# Hybrid Quantum CNN - Dual Encoding
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
        self.dropout = nn.Dropout(0.4)

        # Projection layers
        self.fc_angle = nn.Linear(128 * 4 * 4, 8)
        self.fc_amp   = nn.Linear(128 * 4 * 4, 256)

        # Quantum layers
        self.q_angle = quantum_layer_angle
        self.q_amp   = quantum_layer_amp

        # Fusion classifier → 4 classes
        self.fc_final = nn.Linear(16, 4)  # 8 angle + 8 amp = 16

    def forward(self, x):
        x = self.conv(x)
        features = self.dropout(self.flatten(x))

        # Angle branch
        angle_in  = torch.tanh(self.fc_angle(features))
        q_angle_out = self.q_angle(angle_in)

        # Amplitude branch
        amp_in    = torch.tanh(self.fc_amp(features))
        q_amp_out = self.q_amp(amp_in)

        # Fusion
        fused = torch.cat((q_angle_out, q_amp_out), dim=1)
        out = self.fc_final(fused)
        return out

model = HybridQCNN()