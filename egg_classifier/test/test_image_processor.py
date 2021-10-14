import unittest
from pathlib import Path
from PIL import Image, ImageChops
from egg_classifier.image_processor import ImageSplitter

NUMBER_OF_ROWS: int = 5
NUMBER_OF_COLUMNS: int = 5
INPUT_IMAGE_PATH: str = "./resources/test-dataset/image-split/test-image.jpg"
OUTPUT_IMAGES_DIRECTORY: str = "./resources/test-dataset/image-split/output"
ACTUAL_IMAGES_DIRECTORY: str = "./resources/test-dataset/image-split/actual"


class ImageSplitterTests(unittest.TestCase):
    def test_constructor(self) -> None:
        image_splitter: ImageSplitter = ImageSplitter(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS)
        self.assertEqual(image_splitter.number_of_rows, NUMBER_OF_ROWS)
        self.assertEqual(image_splitter.number_of_columns, NUMBER_OF_COLUMNS)

    def test_split_image(self) -> None:
        image_splitter: ImageSplitter = ImageSplitter(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS)
        output_images: list[list[Image.Image]] = image_splitter.split_image(
            INPUT_IMAGE_PATH
        )

        expected_number_of_output_rows: int = image_splitter.number_of_rows
        expected_number_of_output_columns: int = image_splitter.number_of_columns
        actual_number_of_output_rows: int = len(output_images)
        self.assertEqual(actual_number_of_output_rows, expected_number_of_output_rows, "Invalid Number of Rows in Output")
        for row in output_images:
            actual_number_of_output_columns: int = len(row)
            self.assertEqual(actual_number_of_output_columns, expected_number_of_output_columns, "Invalid Number of Columns in Output")

        path = Path(OUTPUT_IMAGES_DIRECTORY)
        path.mkdir(exist_ok=True)
        for row_counter, row in enumerate(output_images):
            for column_counter, column in enumerate(row):
                column.save(
                    f"{OUTPUT_IMAGES_DIRECTORY}/test-{row_counter}-{column_counter}.jpg",
                    quality=100,
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
                    self.assertIsNone(difference.getbbox(), f"Images Not Equal: {file_name}")


if __name__ == "__main__":
    unittest.main()
