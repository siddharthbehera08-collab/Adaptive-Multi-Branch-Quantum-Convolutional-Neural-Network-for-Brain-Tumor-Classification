model = HybridQCNN()

############################################
# Training Setup
############################################
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

############################################
# Training
############################################
for epoch in range(15):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_acc = 100 * correct / total
    scheduler.step(running_loss)
    print(f"Epoch {epoch+1}/15 | Loss: {running_loss/len(trainloader):.4f} | Train Acc: {train_acc:.2f}%")
    
    
    
    
    
    
    model = HybridQCNN()

############################################
# Training Setup
############################################
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0005)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=2
)

############################################
# Training
############################################
for epoch in range(15):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in trainloader:
        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    scheduler.step(running_loss)

    print(
        f"Epoch {epoch+1}/15 | "
        f"Loss: {running_loss/len(trainloader):.4f} | "
        f"Train Acc: {train_acc:.2f}%"
    )
    
    
    
    
    
    
    
    ############################################
# Training (30 Epochs)
############################################
for epoch in range(30):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in trainloader:
        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    scheduler.step(running_loss)

    print(
        f"Epoch {epoch+1}/30 | "
        f"Loss: {running_loss/len(trainloader):.4f} | "
        f"Train Acc: {train_acc:.2f}%"
    )