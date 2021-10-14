from PIL import Image


class ImageSplitter:
    def __init__(self, number_of_rows: int, number_of_columns: int) -> None:
        self.number_of_rows = number_of_rows
        self.number_of_columns = number_of_columns

    def split_image(self, image_path: str) -> list[list[Image.Image]]:
        with Image.open(image_path) as input_image:
            image_width: int = input_image.width
            image_height: int = input_image.height
            split_width: float = image_width / self.number_of_columns
            splilt_height: float = image_width / self.number_of_rows

            result: list[list[Image.Image]] = []
            y: float = 0
            while y < image_height:
                row: list[Image.Image] = []
                x: float = 0
                while x < image_width:
                    right: float = x + splilt_height
                    bottom: float = y + split_width
                    box = (x, y, right, bottom)
                    sub_image: Image.Image = input_image.crop(box)
                    row.append(sub_image)
                    x += split_width
                y += splilt_height
                result.append(row)

            return result


class ImageDrawer:
    def __init__(self) -> None:
        pass
