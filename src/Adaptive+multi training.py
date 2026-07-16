model = AdaptiveMultiBranchQCNN()

# ============================================================
# Training
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=2, verbose=True
)

best_acc = 0.0
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (pred == labels).sum().item()

    train_acc = 100 * correct / total
    scheduler.step(running_loss)

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Loss: {running_loss/len(trainloader):.4f} | "
          f"Train Acc: {train_acc:.2f}%")
