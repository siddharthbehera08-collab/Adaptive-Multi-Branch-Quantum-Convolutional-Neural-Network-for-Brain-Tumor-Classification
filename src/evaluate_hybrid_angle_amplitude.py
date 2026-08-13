import torch
from hybrid_angle_amplitude_qcnn import HybridQCNN, testloader

model = HybridQCNN()

############################################
# Evaluation
############################################
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Hybrid Angle-Amplitude QCNN Test Accuracy: {accuracy:.2f}%")
