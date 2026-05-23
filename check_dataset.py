import os

train_path = "dataset/train"

print("\nChecking Dataset\n")

for folder in os.listdir(train_path):

    folder_path = os.path.join(train_path, folder)

    if os.path.isdir(folder_path):

        count = len(os.listdir(folder_path))

        print(f"Class {folder}: {count} images")