import unittest
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from egg_classifier.classifier.histogram_classifier import HistogramClassifier
from egg_classifier.image_processor import ImageSplitter

tf.autograph.set_verbosity(3)
DATASET_PATH = "resources/test-dataset/test-data"
ROOT_MODEL_PATH = "resources/test-dataset/test-models"
MODEL_PATH = "resources/test-dataset/test-models/histogram"
IMAGE_PATH = "resources/test-dataset/test-eggs.jpg"
IMAGE_SIZE = (128, 128)


class HistogramClassifierTests(unittest.TestCase):
    def test_load_dataset(self) -> None:
        train_dataset, test_dataset, actual_class_names = HistogramClassifier.load_dataset(
            DATASET_PATH, IMAGE_SIZE, batch_size=1)
        (train_data, train_labels) = train_dataset
        (test_data, test_labels) = test_dataset
        expected_class_names = sorted(os.listdir(DATASET_PATH))
        actual_class_names = sorted(actual_class_names)

        # class names should match
        self.assertEqual(actual_class_names,
                         expected_class_names, "Classes do not match.")

        # number of images should match
        expected_number_of_images = 0
        for class_name in expected_class_names:
            expected_number_of_images += len(
                os.listdir(f"{DATASET_PATH}/{class_name}"))
        train_dataset_number_of_images = len(train_data)
        test_dataset_number_of_images = len(test_data)
        actual_dataset_number_of_images = train_dataset_number_of_images + \
            test_dataset_number_of_images
        print(actual_dataset_number_of_images)
        self.assertEqual(actual_dataset_number_of_images,
                         expected_number_of_images, "Number of images do not match.")

    def test_train(self) -> None:
        if not os.path.exists(ROOT_MODEL_PATH):
            os.mkdir(ROOT_MODEL_PATH)
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)
        model, history = HistogramClassifier.train(
            IMAGE_SIZE, number_of_epochs=1, dataset_path=DATASET_PATH)

        # input shapes should match
        actual_input_shape = model.layers[0].output_shape
        expected_input_shape = (None, 768)
        self.assertEqual(*actual_input_shape,
                         expected_input_shape, "Invalid input shape.")

        # output shapes should match
        actual_output_shape = model.layers[-1].output_shape
        expected_output_shape = (None, 1)
        self.assertEqual(actual_output_shape,
                         expected_output_shape, "Invalid output shape.")

    def test_load_model(self) -> None:
        if not os.path.exists(ROOT_MODEL_PATH):
            os.mkdir(ROOT_MODEL_PATH)
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)
        model, history = HistogramClassifier.train(
            IMAGE_SIZE, number_of_epochs=1, dataset_path=DATASET_PATH, save_path=MODEL_PATH)
        classifier = HistogramClassifier(MODEL_PATH, IMAGE_SIZE)
        model = classifier.model

        # input shapes should match
        actual_input_shape = model.layers[0].output_shape
        expected_input_shape = (None, 768)
        self.assertEqual(*actual_input_shape,
                         expected_input_shape, "Invalid input shape.")

        # output shapes should match
        actual_output_shape = model.layers[-1].output_shape
        expected_output_shape = (None, 1)
        self.assertEqual(actual_output_shape,
                         expected_output_shape, "Invalid output shape.")

    def test_predict(self) -> None:
        if not os.path.exists(ROOT_MODEL_PATH):
            os.mkdir(ROOT_MODEL_PATH)
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)
        model, history = HistogramClassifier.train(
            IMAGE_SIZE, number_of_epochs=1, dataset_path=DATASET_PATH, save_path=MODEL_PATH)
        classifier = HistogramClassifier(MODEL_PATH)

        with Image.open(IMAGE_PATH) as image:
            image_ndarray = np.array(image)
        image_splitter = ImageSplitter(4, 6)
        images = image_splitter.split_image(image_ndarray)
        expected_number_of_predictions = images.shape[0]

        predictions = classifier.predict(images)
        actual_number_of_predictions = len(predictions)
        self.assertEqual(actual_number_of_predictions,
                         expected_number_of_predictions, "Invalid number of predictions.")


if __name__ == "__main__":
    unittest.main()
