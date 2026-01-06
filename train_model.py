import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Define the dataset folder
DATASET_PATH = "C:/ml project"

# Load the dataset
csv_path = os.path.join(DATASET_PATH, "idrid_labels.csv")
df = pd.read_csv(csv_path)

# Ensure the 'id_code' column exists
if 'id_code' not in df.columns:
    raise KeyError("Error: 'id_code' column not found in dataset.")

# Construct the full image path from 'id_code'
df['image_path'] = df['id_code'].apply(lambda x: os.path.join(DATASET_PATH, f"{x}.jpeg"))

# Display some images with labels
fig, axes = plt.subplots(1, 5, figsize=(15, 5))

for i, row in enumerate(df.itertuples()):
    if i >= 5:  # Show only 5 images
        break
    img_path = row.image_path
    label = row.diagnosis  # Update this if the label column is different

    if os.path.exists(img_path):  # Check if image exists
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for proper display
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")
    else:
        print(f"Warning: Image not found - {img_path}")

plt.tight_layout()
plt.show()
