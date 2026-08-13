import torch
import tensorflow as tf
from baselines import build_baseline_cnn
from reduction import test_ds
from quantum_models import HybridQCNN, testloader

############################################
# Baseline CNN Evaluation
############################################
model = build_baseline_cnn()

test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {test_accuracy:.4f}")

############################################
# Transfer Learning Results
############################################
# NOTE: results is populated during train.py's transfer learning loop.
# Running evaluate.py standalone starts with an empty results dict.
# Run train.py first in the same session to populate it.
results = {}

print("\n========== Transfer Learning Results ==========")
for model_name, acc in results.items():
    print(f"{model_name}: {acc * 100:.2f}%")

############################################
# Quantum Model Evaluation
############################################
model = HybridQCNN()

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f"\nQuantum Model Test Accuracy: {accuracy:.2f}%")
