import os
from PIL import Image
import numpy as np
from egg_classifier.image_processor import flip_vertical, flip_horizontal

DATASET_PATH = "resources/dataset"

if __name__ == "__main__":
    dataset_directories = os.listdir(DATASET_PATH)
    for directory in dataset_directories:
        directory_path = f"{DATASET_PATH}/{directory}"
        for file in os.listdir(directory_path):
            file_path = f"{directory_path}/{file}"
            file_name = file.split(".")[0]
            with Image.open(file_path) as image:
                image_ndarray = np.array(image)
            if not file_name.endswith("_flipped_horizontal"):
                image_flipped_vertical = flip_vertical(image_ndarray)
                image_flipped_vertical = Image.fromarray(
                    image_flipped_vertical)
                image_flipped_vertical.save(
                    f"{directory_path}/{file_name}_flipped_vertical.jpg")
            else: print(f"Skipping {file_name}")
            if file_name.endswith("_flipped_horizontal"):
                image_flipped_horizontal = flip_horizontal(image_ndarray)
                image_flipped_horizontal = Image.fromarray(
                    image_flipped_horizontal)
                image_flipped_horizontal.save(
                    f"{directory_path}/{file_name}_flipped_horizontal.jpg")
            else: print(f"Skipping {file_name}")
            print(f"Augmented {file_path}")
