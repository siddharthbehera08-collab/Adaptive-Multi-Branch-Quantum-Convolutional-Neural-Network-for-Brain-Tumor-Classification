test_loss, test_accuracy = model.evaluate(test_ds)

print("Test Accuracy:", test_accuracy)








print("\nModel Comparison")

for model_name, acc in results.items():
    print(model_name, ":", round(acc * 100, 2), "%")
    
    
    
    
    
    
    
    
    
    
    
    
    
import torch

############################################
# Testing
############################################
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

print(f"Test Accuracy: {accuracy:.2f}%")  












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

print(f"Test Accuracy: {accuracy:.2f}%")













import torch

############################################
# Testing
############################################

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

print(f"Test Accuracy: {accuracy:.2f}%")