import matplotlib.pyplot as plt
import torch
import pennylane as qml

# Sample inputs and weights for drawing
sample_inputs = torch.zeros(n_qubits)
sample_weights = torch.zeros(6, n_qubits, 3)

# Draw the quantum circuit
fig, ax = qml.draw_mpl(quantum_circuit)(
    sample_inputs,
    sample_weights
)

plt.title("Quantum Circuit - 8 Qubit QCNN")
plt.show()

# Save the figure
fig.savefig(
    "quantum_circuit.png",
    dpi=300,
    bbox_inches="tight"
)



print(
    qml.draw(quantum_circuit)(
        torch.randn(n_qubits),
        torch.randn(6, n_qubits, 3)
    )
)
















import matplotlib.pyplot as plt
from matplotlib.patches import Patch

############################################
# Test Accuracy Comparison
############################################

models = [
    "Classical CNN",
    "ResNet50",
    "VGG16",
    "MobileNetV2",
    "InceptionV3",
    "Dual Encoding QCNN",
    "Dual Quantum Fusion QCNN"
]

accuracies = [
    92.00,   # Classical CNN
    71.62,   # ResNet50
    75.71,   # VGG16
    83.46,   # MobileNetV2
    78.32,   # InceptionV3
    86.19,   # Dual Encoding QCNN
    86.65    # Two Quantum Circuits + Fusion QCNN
]

colors = [
    "#4C72B0",
    "#4C72B0",
    "#4C72B0",
    "#4C72B0",
    "#4C72B0",
    "#DD8452",
    "#C44E52"
]

############################################
# Plot
############################################

plt.figure(figsize=(13,6))

bars = plt.bar(
    models,
    accuracies,
    color=colors,
    edgecolor="black",
    linewidth=0.8
)

for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        acc + 0.5,
        f"{acc:.2f}%",
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

plt.axhline(
    y=90,
    color="red",
    linestyle="--",
    linewidth=1.2,
    label="90% Accuracy"
)

plt.title(
    "Performance Comparison of Classical and Hybrid Quantum Models\nBrain Tumor MRI Classification",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Models", fontsize=12)
plt.ylabel("Test Accuracy (%)", fontsize=12)

plt.xticks(rotation=15)

plt.ylim(65,100)

plt.grid(axis="y", linestyle="--", alpha=0.35)

legend_elements = [
    Patch(facecolor="#4C72B0", label="Classical Models"),
    Patch(facecolor="#DD8452", label="Hybrid Quantum Models"),
    Patch(facecolor="#C44E52", label="Best Proposed Model")
]

plt.legend(handles=legend_elements, loc="lower right")

plt.tight_layout()

plt.savefig(
    "accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

############################################
# Summary Table
############################################

print("\n========== Model Comparison ==========\n")

print(f"{'Model':<35}{'Accuracy':>10}{'Type':>18}")
print("-"*65)

for model, acc in zip(models, accuracies):

    if "QCNN" in model:
        model_type = "Hybrid Quantum"
    else:
        model_type = "Classical"

    print(f"{model:<35}{acc:>9.2f}%{model_type:>18}")

print("-"*65)

best_classical = max(accuracies[:5])
best_quantum = max(accuracies[5:])

print(f"\nBest Classical Model : {best_classical:.2f}%")
print(f"Best Quantum Model   : {best_quantum:.2f}%")
print(f"Accuracy Gap         : {best_classical-best_quantum:.2f}%")