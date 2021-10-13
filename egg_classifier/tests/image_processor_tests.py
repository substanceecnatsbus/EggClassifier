import unittest
from PIL import Image, ImageChops
from egg_classifier.image_processor import ImageSplitter

NUMBER_OF_ROWS: int = 5
NUMBER_OF_COLUMNS: int = 5
INPUT_IMAGE: str = "./dataset/test/image-split/test.jpg"
OUTPUT_IMAGES_DIRECTORY: str = "./dataset/test/image-split/output"
ACTUAL_IMAGES_DIRECTORY: str = "./dataset/test/image-split/actual"


class ImageSplitterTests(unittest.TestCase):
    def test_constructor(self) -> None:
        image_splitter: ImageSplitter = ImageSplitter(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS)
        self.assertEqual(image_splitter.number_of_rows, NUMBER_OF_ROWS)
        self.assertEqual(image_splitter.number_of_columns, NUMBER_OF_COLUMNS)

    def test_split_image(self) -> None:
        image_splitter: ImageSplitter = ImageSplitter(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS)
        with Image.open(INPUT_IMAGE) as input_image:
            output_images: list[list[Image.Image]] = image_splitter.split_image(
                input_image
            )
            for row_counter, row_counter in enumerate(output_images):
                for column_counter, column in enumerate(row_counter):
                    column.save(
                        f"{OUTPUT_IMAGES_DIRECTORY}/test-{row_counter}-{column_counter}.jpg"
                    )
        for row_counter in range(5):
            for column_counter in range(5):
                file_name: str = f"test-{row_counter}-{column_counter}.jpg"
                with Image.open(
                    f"{OUTPUT_IMAGES_DIRECTORY}/{file_name}"
                ) as output_image, Image.open(
                    f"{ACTUAL_IMAGES_DIRECTORY}/{file_name}"
                ) as actual_image:
                    difference: Image.Image = ImageChops.difference(
                        output_image, actual_image
                    )
                    self.assertIsNone(difference.getbbox(), f"Image: ${file_name}")


if __name__ == "__main__":
    unittest.main()
