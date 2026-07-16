# ============================================================
# Evaluation
# ============================================================
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in testloader:
        _, pred = torch.max(model(images), 1)
        total   += labels.size(0)
        correct += (pred == labels).sum().item()

print(f"\nTest Accuracy: {100*correct/total:.2f}%")

# ============================================================
# Inspect what the adaptive encoding learned
# (shows which features each circuit decided to amplify)
# ============================================================
print("\n--- Learned encoding weights (alpha_i) ---")
for name, layer in [("Edge", model.q_edge),
                    ("Texture", model.q_texture),
                    ("Global", model.q_global)]:
    alphas = layer.encoding_weights.detach().numpy()
    print(f"{name:8s}: {[f'{a:.3f}' for a in alphas]}")