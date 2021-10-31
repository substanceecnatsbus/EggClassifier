import os
from PIL import Image
import numpy as np
from egg_classifier.image_processor import ImageSplitter

RAW_DATASET_PATH = "resources/raw-dataset"
CROPPED_DATASET_PATH = "resources/cropped-dataset"
OFFSET_X = -10
OFFSET_Y = 0

if __name__ == "__main__":
    if not os.path.exists(CROPPED_DATASET_PATH):
        os.mkdir(CROPPED_DATASET_PATH)

    image_splitter = ImageSplitter(4, 6, OFFSET_X, OFFSET_Y)
    for file in os.listdir(RAW_DATASET_PATH):
        file_name = file.split(".")[0]
        with Image.open(f"{RAW_DATASET_PATH}/{file}") as fin:
            image = np.array(fin)
            images = image_splitter.split_image(image)
            number_of_images = images.shape[0]
            for i in range(number_of_images):
                cropped_image = images[i, :]
                cropped_image = Image.fromarray(cropped_image)
                cropped_image.save(
                    f"{CROPPED_DATASET_PATH}/{file_name}_{i}.jpg")
