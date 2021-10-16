import unittest
from PIL import Image
import numpy as np
from egg_classifier.image_processor import ImageSplitter

NUMBER_OF_ROWS: int = 5
NUMBER_OF_COLUMNS: int = 5
INPUT_IMAGE_PATH: str = "./resources/test-dataset/test-image.jpg"


class ImageSplitterTests(unittest.TestCase):
    def test_constructor(self) -> None:
        image_splitter = ImageSplitter(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS)
        self.assertEqual(image_splitter.number_of_rows, NUMBER_OF_ROWS)
        self.assertEqual(image_splitter.number_of_columns, NUMBER_OF_COLUMNS)

    def test_split_image(self) -> None:
        input_image: Image.Image = None
        try:
            # get ImageSplitter.split_image output
            image_splitter = ImageSplitter(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS)
            input_image = Image.open(INPUT_IMAGE_PATH)
            input_image_ndarray = np.array(input_image, dtype=np.uint8)
            output_images = image_splitter.split_image(input_image_ndarray)

            # get input image properties
            input_image_width = input_image.width
            input_image_height = input_image.height
            crop_width = int(input_image_width / image_splitter.number_of_columns)
            crop_height = int(input_image_height / image_splitter.number_of_rows)
            actual_dimensions = output_images.shape

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
            while y_start < input_image_height:
                y_end = y_start + crop_height
                x_start = 0
                while x_start < input_image_width:
                    x_end = x_start + crop_width
                    expected_image = input_image_ndarray[y_start:y_end, x_start:x_end]
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


if __name__ == "__main__":
    unittest.main()
