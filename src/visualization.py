import matplotlib.pyplot as plt
import torch
import pennylane as qml
from quantum_models import n_qubits, quantum_circuit

############################################
# Quantum Circuit Diagram
############################################
sample_weights = torch.zeros(6, n_qubits, 3)

fig, ax = qml.draw_mpl(quantum_circuit)(
    torch.zeros(n_qubits),
    sample_weights
)

plt.title("Hybrid QCNN - Quantum Circuit")
plt.tight_layout()
plt.savefig("quantum_circuit_diagram.png", dpi=150, bbox_inches="tight")
plt.show()

############################################
# Model Accuracy Bar Chart
############################################
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

model_names = [
    "Baseline CNN",
    "ResNet50",
    "VGG16",
    "MobileNetV2",
    "EfficientNetB0",
    "InceptionV3",
    "Hybrid QCNN"
]

accuracies = [93.0, 71.6, 75.7, 83.5, 53.4, 78.3, 77.2]

colors = [
    "#4e79a7",
    "#f28e2b", "#f28e2b", "#f28e2b", "#f28e2b", "#f28e2b",
    "#e15759"
]

fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.bar(model_names, accuracies, color=colors, edgecolor="black", linewidth=0.8)

for bar, acc in zip(bars, accuracies):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{acc:.1f}%",
        ha="center", va="bottom", fontsize=10, fontweight="bold"
    )

ax.set_ylim(0, 105)
ax.set_ylabel("Test Accuracy (%)", fontsize=12)
ax.set_title("Model Comparison: Brain Tumor Classification", fontsize=14, fontweight="bold")
ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)

legend_elements = [
    Patch(facecolor="#4e79a7", edgecolor="black", label="Classical CNN (Scratch)"),
    Patch(facecolor="#f28e2b", edgecolor="black", label="Transfer Learning"),
    Patch(facecolor="#e15759", edgecolor="black", label="Hybrid QCNN"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

plt.tight_layout()
plt.savefig("model_accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
