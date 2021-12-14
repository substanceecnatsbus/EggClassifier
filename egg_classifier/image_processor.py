from typing import Dict, List
from PIL import Image, ImageDraw, ImageFont
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
                    y_start: y_end + offset_y,
                    x_start: x_end + offset_x,
                ]
                images.append(cropped_image)
                x_start += crop_width
            y_start += crop_height

        result = np.stack(images)
        return result


class ImageDrawer:
    def __init__(self, number_of_rows: int, number_columns: int, width: int,
                 colors: Dict[str, str], font: str, font_size: int) -> None:
        self.number_of_rows = number_of_rows
        self.number_of_columns = number_columns
        self.width = width
        self.colors = colors
        self.font = font
        self.font_size = font_size

    def draw(self, image_np: np.ndarray, labels: List[str]) -> np.ndarray:
        assert self.number_of_rows * self.number_of_columns == len(labels)
        image = Image.fromarray(image_np)
        input_image_width = image.width
        input_image_height = image.height
        crop_width = input_image_width // self.number_of_columns
        crop_height = input_image_height // self.number_of_rows
        font = ImageFont.truetype(self.font, self.font_size)

        d = ImageDraw.Draw(image)
        label_counter = 0
        for y_start in range(0, crop_height * self.number_of_rows, crop_height):
            y_end = y_start + crop_height
            for x_start in range(0, crop_width * self.number_of_columns, crop_width):
                label = labels[label_counter]
                color = self.colors[label]
                x_end = x_start + crop_width
                center_y = y_start + (y_end - y_start) // 2
                center_x = x_start + (x_end - x_start) // 2
                if(label == "fertile"):
                    d.line([(center_x - self.width * 2.5, center_y - self.width * 2),
                           (center_x - self.width * 0.5, center_y + self.width)], color, self.width)
                    d.line([(center_x - self.width * 0.5, center_y + self.width),
                           (center_x + self.width * 4.5, center_y - self.width * 2)], color, self.width)
                else:
                    d.line([(center_x - self.width * 2, center_y - self.width * 3),
                           (center_x + self.width * 2, center_y + self.width * 3)], color, self.width)
                    d.line([(center_x + self.width * 2, center_y - self.width * 3),
                           (center_x - self.width * 2, center_y + self.width * 3)], color, self.width)
                d.text((center_x - self.width - self.font_size * 0.95, center_y -
                       self.width - self.font_size * 2), label, color, font=font)
                label_counter += 1

        return np.array(image)


def flip_vertical(input_image: np.ndarray) -> np.ndarray:
    result = np.flip(input_image, axis=0)
    return result


def flip_horizontal(input_image: np.ndarray):
    result = np.flip(input_image, axis=1)
    return result
