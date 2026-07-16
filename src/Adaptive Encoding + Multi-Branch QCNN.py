# ============================================================
# Adaptive Encoding + Multi-Branch QCNN
# Two ideas combined:
#   1. Adaptive Encoding: theta_i = alpha_i * x_i
#      The circuit learns HOW to encode, not just what to do after.
#   2. Multi-Branch: separate quantum circuits for edge, texture,
#      and global features — each specializes on one scale.
# ============================================================import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
import numpy as np

# ---------- TF → NumPy ----------
def tf_to_numpy(ds):
    X, Y = [], []
    for x, y in ds:
        X.append(x.numpy())
        Y.append(y.numpy())
    return np.concatenate(X), np.concatenate(Y)

X_train, y_train = tf_to_numpy(train_ds)
X_test,  y_test  = tf_to_numpy(test_ds)

X_train = torch.tensor(X_train).permute(0,3,1,2).float()
X_test  = torch.tensor(X_test ).permute(0,3,1,2).float()
y_train = torch.tensor(y_train).long()
y_test  = torch.tensor(y_test ).long()

trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
testloader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=16)

# ============================================================
# Quantum circuit with adaptive encoding
#
# Standard: theta_i = x_i  (fixed — circuit can't adjust encoding)
# Adaptive: theta_i = alpha_i * x_i  (alpha_i are learnable parameters)
#
# Why it helps: the circuit learns which input features to amplify
# or suppress before rotation. Low-signal features get small alpha,
# high-signal features get large alpha. Fixed encoding treats all 8
# inputs equally — wasteful.
# ============================================================

n_qubits = 8
n_layers = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def adaptive_circuit(inputs, encoding_weights, circuit_weights):
    # --- Adaptive encoding layer ---
    # encoding_weights = alpha_i, shape (n_qubits,)
    # Each qubit gets: RY(alpha_i * x_i) instead of RY(x_i)
    for i in range(n_qubits):
        qml.RY(encoding_weights[i] * inputs[i], wires=i)

    # --- Variational layers ---
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RY(circuit_weights[l, i, 0], wires=i)
            qml.RZ(circuit_weights[l, i, 1], wires=i)
        # Circular entanglement
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {
    "encoding_weights": (n_qubits,),          # the alpha_i
    "circuit_weights":  (n_layers, n_qubits, 2)
}

# Three separate circuits — each branch has its own trainable weights
q_edge    = qml.qnn.TorchLayer(adaptive_circuit, weight_shapes)
q_texture = qml.qnn.TorchLayer(adaptive_circuit, weight_shapes)
q_global  = qml.qnn.TorchLayer(adaptive_circuit, weight_shapes)

# ============================================================
# Model: CNN extracts 3 feature scales, each goes through its
# own adaptive quantum circuit, outputs are concatenated and
# classified by a small MLP head.
#
# Edge     (conv1, 32ch)  → shallow features = low-level spatial
# Texture  (conv2, 64ch)  → mid features = patterns and structures
# Global   (conv3, 128ch) → deep features = semantic content
# ============================================================

class AdaptiveMultiBranchQCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Shared CNN backbone ---
        # Input: [B, 3, 128, 128]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),            # → [B, 32, 64, 64]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),            # → [B, 64, 32, 32]
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4)) # → [B, 128, 4, 4]
        )

        # Pool edge/texture down to 4×4 before flattening
        self.pool_e = nn.AdaptiveAvgPool2d((4,4))  # 32*4*4 = 512
        self.pool_t = nn.AdaptiveAvgPool2d((4,4))  # 64*4*4 = 1024

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)

        # --- Projection layers (→ 8D quantum input) ---
        self.proj_edge    = nn.Linear(32  * 4 * 4, 8)
        self.proj_texture = nn.Linear(64  * 4 * 4, 8)
        self.proj_global  = nn.Linear(128 * 4 * 4, 8)

        self.bn_edge    = nn.BatchNorm1d(8)
        self.bn_texture = nn.BatchNorm1d(8)
        self.bn_global  = nn.BatchNorm1d(8)

        # --- Quantum branches ---
        self.q_edge    = q_edge
        self.q_texture = q_texture
        self.q_global  = q_global

        # --- Fusion head: 3 × 8 = 24 → 4 classes ---
        self.fc_fusion = nn.Sequential(
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        # Extract features at three scales
        e = self.conv1(x)       # edge
        t = self.conv2(e)       # texture
        g = self.conv3(t)       # global

        # Pool → flatten → dropout
        e_f = self.dropout(self.flatten(self.pool_e(e)))
        t_f = self.dropout(self.flatten(self.pool_t(t)))
        g_f = self.dropout(self.flatten(g))

        # Project to 8D, normalize to [-1, 1] for angle encoding
        e_in = torch.tanh(self.bn_edge(self.proj_edge(e_f)))
        t_in = torch.tanh(self.bn_texture(self.proj_texture(t_f)))
        g_in = torch.tanh(self.bn_global(self.proj_global(g_f)))

        # Adaptive quantum processing
        e_q = self.q_edge(e_in)       # [B, 8]
        t_q = self.q_texture(t_in)    # [B, 8]
        g_q = self.q_global(g_in)     # [B, 8]

        # Fuse and classify
        fused = torch.cat([e_q, t_q, g_q], dim=1)  # [B, 24]
        return self.fc_fusion(fused)


model = AdaptiveMultiBranchQCNN()