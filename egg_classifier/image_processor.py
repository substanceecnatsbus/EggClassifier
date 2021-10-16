import numpy as np


class ImageSplitter:
    def __init__(self, number_of_rows: int, number_of_columns: int) -> None:
        self.number_of_rows = number_of_rows
        self.number_of_columns = number_of_columns

    def split_image(self, image: np.ndarray) -> np.ndarray:
        image_dimensions = image.shape
        image_width = image_dimensions[1]
        image_height = image_dimensions[0]
        crop_width = int(image_width / self.number_of_columns)
        crop_height = int(image_height / self.number_of_rows)

        images = []
        y_start = 0
        while y_start < image_height:
            y_end = y_start + crop_height
            x_start = 0
            while x_start < image_width:
                x_end = x_start + crop_width
                cropped_image = image[y_start:y_end, x_start:x_end]
                cropped_image = cropped_image.tolist()
                images.append(cropped_image)
                x_start += crop_width
            y_start += crop_height

        result = np.array(images)
        return result


class ImageDrawer:
    def __init__(self) -> None:
        pass
