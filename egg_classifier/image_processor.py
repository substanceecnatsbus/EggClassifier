import numpy as np


class ImageSplitter:
    def __init__(
        self,
        number_of_rows: int,
        number_of_columns: int,
        offset_x_percent: int = 0,
        offset_y_percent: int = 0,
    ) -> None:
        self.number_of_rows = number_of_rows
        self.number_of_columns = number_of_columns
        self.offset_x_percent = offset_x_percent
        self.offset_y_percent = offset_y_percent

    def split_image(self, image: np.ndarray) -> np.ndarray:
        image_dimensions = image.shape
        image_width = image_dimensions[1]
        image_height = image_dimensions[0]
        crop_width = image_width // self.number_of_columns
        crop_height = image_height // self.number_of_rows
        offset_x = crop_width * self.offset_x_percent // 100
        offset_y = crop_height * self.offset_y_percent // 100

        images = []
        y_start = 0
        while y_start < crop_height * self.number_of_rows:
            y_end = y_start + crop_height
            if y_end + offset_y > image_height:
                raise Exception(
                    "The offset in the y-axis is going over the image height."
                )
            x_start = 0
            while x_start < crop_width * self.number_of_columns:
                x_end = x_start + crop_width
                if x_end + offset_x > image_width:
                    raise Exception(
                        "The offset in the x-axis is going over the image width."
                    )
                cropped_image = image[
                    y_start : y_end + offset_y,
                    x_start : x_end + offset_x,
                ]
                images.append(cropped_image)
                x_start += crop_width
            y_start += crop_height

        result = np.stack(images)
        return result


class ImageDrawer:
    def __init__(self) -> None:
        pass
