import unittest
import os
import tensorflow as tf
from egg_classifier.classifier.mobilenetv2_classifier import Mobilenetv2Classifier
from egg_classifier.test.test_image_processor import IMAGE_SPLITTER_INPUT_IMAGE_PATH

tf.autograph.set_verbosity(3)
DATASET_PATH = "resources/test-dataset/test-data"
ROOT_MODEL_PATH = "resources/test-dataset/test-models"
MODEL_PATH = "resources/test-dataset/test-models/mobilenetv2"
IMAGE_SIZE = (128, 128)


class MobilenetV2ClassifierTests(unittest.TestCase):
    def test_load_dataset(self) -> None:
        train_dataset, test_dataset, actual_class_names = Mobilenetv2Classifier.load_dataset(
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
        if not os.path.exists(ROOT_MODEL_PATH):
            os.mkdir(ROOT_MODEL_PATH)
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)
        model, history = Mobilenetv2Classifier.train(
            (IMAGE_SIZE[0], IMAGE_SIZE[1], 3), number_of_epochs=1,
            dataset_path=DATASET_PATH
        )

        # input shapes should match
        actual_input_shape = model.layers[0].output_shape
        expected_input_shape = (None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
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
        model, history = Mobilenetv2Classifier.train(
            (IMAGE_SIZE[0], IMAGE_SIZE[1], 3), number_of_epochs=1,
            dataset_path=DATASET_PATH, save_path=MODEL_PATH
        )
        classifier = Mobilenetv2Classifier(MODEL_PATH)
        model = classifier.model

        # input shapes should match
        actual_input_shape = model.layers[0].output_shape
        expected_input_shape = (None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        self.assertEqual(*actual_input_shape,
                         expected_input_shape, "Invalid input shape.")

        # output shapes should match
        actual_output_shape = model.layers[-1].output_shape
        expected_output_shape = (None, 1)
        self.assertEqual(actual_output_shape,
                         expected_output_shape, "Invalid output shape.")


if __name__ == "__main__":
    unittest.main()
