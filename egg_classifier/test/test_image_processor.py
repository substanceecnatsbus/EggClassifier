import unittest
from PIL import Image
import numpy as np
from egg_classifier.image_processor import ImageSplitter, flip_vertical, flip_horizontal

NUMBER_OF_ROWS: int = 4
NUMBER_OF_COLUMNS: int = 6
OFFSET_X_PERCENT: int = -10
OFFSET_Y_PERCENT: int = 0
IMAGE_SPLITTER_INPUT_IMAGE_PATH: str = "./resources/test-dataset/test-eggs.jpg"
IMAGE_AUGMENTATION_INPUT_IMAGE_PATH: str = "./resources/test-dataset/test-egg.jpg"


class ImageSplitterTests(unittest.TestCase):
    def test_constructor(self) -> None:
        image_splitter = ImageSplitter(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS)
        self.assertEqual(image_splitter.number_of_rows, NUMBER_OF_ROWS)
        self.assertEqual(image_splitter.number_of_columns, NUMBER_OF_COLUMNS)

    def test_split_image(self) -> None:
        input_image: Image.Image = None
        try:
            # get ImageSplitter.split_image output
            image_splitter = ImageSplitter(
                NUMBER_OF_ROWS, NUMBER_OF_COLUMNS, OFFSET_X_PERCENT, OFFSET_Y_PERCENT
            )
            input_image = Image.open(IMAGE_SPLITTER_INPUT_IMAGE_PATH)
            input_image_ndarray = np.array(input_image, dtype=np.uint8)
            output_images = image_splitter.split_image(input_image_ndarray)

            # get input image properties
            input_image_width = input_image.width
            input_image_height = input_image.height
            crop_width = int(input_image_width /
                             image_splitter.number_of_columns)
            crop_height = int(input_image_height /
                              image_splitter.number_of_rows)
            actual_dimensions = output_images.shape
            offset_x = crop_width * image_splitter.offset_x_percent // 100
            offset_y = crop_height * image_splitter.offset_y_percent // 100

            # number of images should match
            expected_number_of_images = (
                image_splitter.number_of_rows * image_splitter.number_of_columns
            )
            self.assertEqual(
                actual_dimensions[0],
                expected_number_of_images,
                "Invalid number of images.",
            )

            # images should match
            image_counter = 0
            y_start = 0
            while y_start < crop_height * image_splitter.number_of_rows:
                y_end = y_start + crop_height
                x_start = 0
                while x_start < crop_width * image_splitter.number_of_columns:
                    x_end = x_start + crop_width
                    expected_image = input_image_ndarray[
                        y_start: y_end + offset_y, x_start: x_end + offset_x
                    ]
                    actual_image = output_images[image_counter]
                    self.assertTrue(
                        np.allclose(actual_image, expected_image),
                        f"Image {image_counter} does not match.",
                    )
                    x_start += crop_width
                    image_counter += 1
                y_start += crop_height
        except Exception as error:
            self.fail(error)
        finally:
            if input_image is not None:
                input_image.close()


class ImageAugmentationTests(unittest.TestCase):
    def test_flip_vertical(self) -> None:
        input_image: Image.Image = None
        try:
            input_image = Image.open(IMAGE_AUGMENTATION_INPUT_IMAGE_PATH)
            input_image_ndarray = np.array(input_image, np.uint8)
            input_image_shape = input_image_ndarray.shape
            actual_image = flip_vertical(input_image_ndarray)
            actual_image_shape = actual_image.shape

            # shapes should match
            self.assertEqual(
                input_image_shape, actual_image_shape, "Invalid image shape."
            )

            # images should match
            expected_image = np.flip(input_image_ndarray, axis=0)
            self.assertTrue(
                np.allclose(
                    actual_image, expected_image), "Image does not match."
            )
        except Exception as error:
            self.fail(error)
        finally:
            if input_image is not None:
                input_image.close()

    def test_flip_horizontal(self) -> None:
        input_image: Image.Image = None
        try:
            input_image = Image.open(IMAGE_AUGMENTATION_INPUT_IMAGE_PATH)
            input_image_ndarray = np.array(input_image, np.uint8)
            input_image_shape = input_image_ndarray.shape
            actual_image = flip_horizontal(input_image_ndarray)
            actual_image_shape = actual_image.shape

            # shapes should match
            self.assertEqual(
                input_image_shape, actual_image_shape, "Invalid image shape."
            )

            # images should match
            expected_image = np.flip(input_image_ndarray, axis=1)
            self.assertTrue(
                np.allclose(
                    actual_image, expected_image), "Image does not match."
            )
        except Exception as error:
            self.fail(error)
        finally:
            if input_image is not None:
                input_image.close()


if __name__ == "__main__":
    unittest.main()
