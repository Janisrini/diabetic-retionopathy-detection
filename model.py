import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.utils import load_img,img_to_array

# Set dataset path
DATASET_PATH = "idrid_labels"  # Update with your actual dataset path
LABELS_CSV = "idrid_labels.csv"  # Update with your actual labels CSV file

# Load labels
df = pd.read_csv(os.path.join(DATASET_PATH, LABELS_CSV))
print(df.head())

# Count of each class (Diabetic Retinopathy levels)
plt.figure(figsize=(8, 5))
sns.countplot(x=df['diagnosis'], palette="viridis")
plt.title("Class Distribution of Diabetic Retinopathy")
plt.xlabel("Diagnosis (0 = No DR, 4 = Severe DR)")
plt.ylabel("Count")
plt.show()

# Function to visualize random images from each class
def show_images_per_class(df, num_images=5):
    unique_classes = df['diagnosis'].unique()
    fig, axes = plt.subplots(len(unique_classes), num_images, figsize=(15, 10))
    
    for i, class_label in enumerate(unique_classes):
        class_images = df[df['diagnosis'] == class_label]['image_path'].values
        sample_images = np.random.choice(class_images, num_images, replace=False)
        
        for j, img_path in enumerate(sample_images):
            img = load_img(os.path.join(DATASET_PATH, img_path), target_size=(224, 224))
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_title(f"Class {class_label}")

    plt.tight_layout()
    plt.show()

# Call function to display images
show_images_per_class(df)
