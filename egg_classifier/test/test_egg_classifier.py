import unittest
from PIL import Image
import numpy as np
from egg_classifier import ClassifierType, EggClassifier

TEST_IMAGE = "resources/test-dataset/test-eggs.jpg"
NUMBER_OF_ROWS = 4
NUMBER_OF_COLUMNS = 6
OFFSET_X_PERCENT = 0
OFFSET_Y_PERCENT = 0
RADIUS = 7
COLORS = dict([
    ("fertile", "#0000ff"),
    ("infertile", "#ff0000")
])
FONT = "arial.ttf"
FONT_SIZE = 18
CLASSIFIER_TYPE = ClassifierType.HISTOGRAM
MODEL_PATH = "resources/models/histogram"
CLASSES = [
    "fertile",
    "infertile"
]
PREDICTION_THRESHOLD = 0.4
IMAGE_SIZE = [
    128,
    128
]


class EggClassifierTests(unittest.TestCase):
    def test_predict(self) -> None:
        egg_classifier = EggClassifier(
            NUMBER_OF_ROWS, NUMBER_OF_COLUMNS, OFFSET_X_PERCENT,
            OFFSET_Y_PERCENT, RADIUS, COLORS, FONT,
            FONT_SIZE, CLASSIFIER_TYPE, MODEL_PATH,
            CLASSES, PREDICTION_THRESHOLD, IMAGE_SIZE
        )
        with Image.open(TEST_IMAGE) as image:
            actual_image_width = image.width
            actual_image_height = image.height
            image_np = np.array(image)
        output_image_np = egg_classifier.predict(image_np)
        output_image = Image.fromarray(output_image_np)
        expected_image_width = output_image.width
        expected_image_height = output_image.height
        self.assertEqual(
            (actual_image_width, actual_image_height),
            (expected_image_width, expected_image_height),
            "Invalid output image dimensions."
        )
