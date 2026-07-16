from baselines import build_baseline_cnn

model = build_baseline_cnn()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)











from baselines import build_improved_baseline_cnn

model = build_improved_baseline_cnn()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=callbacks
)











callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2
    )
]

results = {}

for name, model_class in models_dict.items():
    print("\nTraining:", name)

    base_model = model_class(
        weights="imagenet",
        include_top=False,
        input_shape=(128,128,3)
    )

    model = build_model(base_model)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks,
        verbose=1
    )

    val_acc = max(history.history["val_accuracy"])
    results[name] = val_acc
    
    
    
    
    
    
    
    
    
    
    
    
    
    
import torch.optim as optim

model = HybridQCNN()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005
)

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
    
    
    
    
    
    
    
    
    
    
    
    
    import torch.optim as optim

model = HybridQCNN()

############################################
# Training Setup
############################################

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005
)

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
    
    
    
    
    
    
    
    
    
    
    
    