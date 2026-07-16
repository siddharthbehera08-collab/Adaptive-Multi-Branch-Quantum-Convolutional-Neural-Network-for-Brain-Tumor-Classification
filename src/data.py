import os
from config import TRAIN_DATASET_PATH, TEST_DATASET_PATH

print("=== TRAINING SET ===")
for cls in os.listdir(TRAIN_DATASET_PATH):
    count = len(os.listdir(os.path.join(TRAIN_DATASET_PATH, cls)))
    print(f"  {cls}: {count} images")

print("\n=== TESTING SET ===")
for cls in os.listdir(TEST_DATASET_PATH):
    count = len(os.listdir(os.path.join(TEST_DATASET_PATH, cls)))
    print(f"  {cls}: {count} images")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    TRAIN_DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    TRAIN_DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)