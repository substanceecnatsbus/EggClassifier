import unittest
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from egg_classifier.classifier import Classifier

DATASET_PATH = "resources/test-dataset/test-data"
IMAGE_SIZE = (128, 128)
INPUT_SHAPE = (128, 128, 3)
tf.autograph.set_verbosity(3)


class ClassifierTests(unittest.TestCase):
    def test_load_dataset(self) -> None:
        train_dataset, test_dataset, actual_class_names = Classifier.load_dataset(
            DATASET_PATH, IMAGE_SIZE, batch_size=1)
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
        train_dataset_number_of_images = tf.data.experimental.cardinality(
            train_dataset)
        test_dataset_number_of_images = tf.data.experimental.cardinality(
            test_dataset)
        actual_dataset_number_of_images = train_dataset_number_of_images + \
            test_dataset_number_of_images
        self.assertEqual(actual_dataset_number_of_images,
                         expected_number_of_images, "Number of images do not match.")

    def test_train(self) -> None:
        model, history = Classifier.train(INPUT_SHAPE, number_of_epochs=1, dataset_path="resources/test-dataset/test-data")

        # input shapes should match
        actual_input_shape = model.layers[0].output_shape
        expected_input_shape = (None, INPUT_SHAPE[0], INPUT_SHAPE[1], 3)
        self.assertEqual(*actual_input_shape,
                         expected_input_shape, "Invalid input shape.")

        # output shapes should match
        actual_output_shape = model.layers[-1].output_shape
        expected_output_shape = (None, 1)
        self.assertEqual(actual_output_shape,
                         expected_output_shape, "Invalid output shape.")


if __name__ == "__main__":
    unittest.main()
