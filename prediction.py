import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define dataset path
DATASET_PATH = "./"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

# Load dataset
df = pd.read_csv(os.path.join(DATASET_PATH, "idrid_labels.csv"))
df.dropna(axis=1, how='all', inplace=True)  # Remove unnamed empty columns

# Ensure column names match
df.rename(columns={"id_code": "image_id", "diagnosis": "label"}, inplace=True)

df["image_path"] = df["image_id"].apply(lambda x: os.path.join(DATASET_PATH, x + ".jpeg"))

# Check if all images exist
df = df[df["image_path"].apply(os.path.exists)]

# Visualizing some images
def visualize_images(df, num_images=5):
    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        img_path = df.iloc[i]["image_path"]
        img = load_img(img_path, target_size=IMAGE_SIZE)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Label: {df.iloc[i]['label']}")
    plt.show()

visualize_images(df)

# Encoding labels
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["label"])

# Train-test split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
train_generator = datagen.flow_from_dataframe(
    train_df, x_col="image_path", y_col="label", target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, class_mode="categorical")
val_generator = datagen.flow_from_dataframe(
    val_df, x_col="image_path", y_col="label", target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, class_mode="categorical")

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(df["label"].unique()), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Save the trained model
model.save("diabetic_retinopathy_model.h5")

print("Model training complete! Model saved as diabetic_retinopathy_model.h5")