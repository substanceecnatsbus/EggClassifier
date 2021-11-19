from enum import Enum, auto
from typing import Dict, Tuple, List
import time
import subprocess
import platform
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from egg_classifier.image_processor import ImageSplitter, ImageDrawer
from egg_classifier.classifier import Classifier
from egg_classifier.classifier.mobilenetv2_classifier import Mobilenetv2Classifier
from egg_classifier.classifier.histogram_classifier import HistogramClassifier

tf.autograph.set_verbosity(0)


class ClassifierType(Enum):
    MOBILENETV2 = auto()
    HISTOGRAM = auto()


class EggClassifier():
    def __init__(self, number_of_rows: int, number_of_columns: int,
                 offset_x_percent: int, offset_y_percent: int,
                 radius: int, colors: Dict[str, str], font: str,
                 font_size: int, classifier_type: ClassifierType, model_path: str,
                 classes: List[str], prediction_threshold: float,
                 image_size: Tuple[int, int] = (128, 128)) -> None:
        assert len(classes) == 2
        assert prediction_threshold >= 0 and prediction_threshold <= 1
        assert len(colors) == 2

        self.__image_splitter: ImageSplitter = ImageSplitter(number_of_rows, number_of_columns,
                                                             offset_x_percent, offset_y_percent)
        self.__image_drawer: ImageDrawer = ImageDrawer(number_of_rows, number_of_columns, radius,
                                                       colors, font, font_size)
        self.__classes = classes
        self.__prediction_threshold = prediction_threshold
        self.__classifier: Classifier = None
        if classifier_type == ClassifierType.MOBILENETV2:
            raise Exception("Mobilenetv2 is depreciated")
            self.__classifier = Mobilenetv2Classifier(model_path, image_size)
        else:
            self.__classifier = HistogramClassifier(model_path, image_size)

    def predict(self, image_np: np.ndarray) -> np.ndarray:
        images = self.__image_splitter.split_image(image_np)
        outputs = self.__classifier.predict(images)
        labels = [self.__classes[0] if x <=
                  self.__prediction_threshold else self.__classes[1] for x in outputs]
        output_image = self.__image_drawer.draw(image_np, labels)
        return output_image


class EggClassifierUI():
    def __init__(self, classifier: EggClassifier, image_size: Tuple[int, int], temp_path: str):
        self.__classifier = classifier
        self.__image_size = image_size
        self.__temp_path = temp_path
        self.__image_tk = None
        self.__initialize()

    def __initialize(self):
        self.__root = Tk()
        self.__root.title("Egg Classifier")
        self.__root.resizable(False, False)
        self.__master = ttk.Frame(self.__root)
        self.__master.grid(row=0, column=0, padx=8, pady=4)
        file_button = ttk.Button(
            self.__master,
            text="Browse",
            command=self.__load_file
        )
        file_button.grid(row=1, column=0, sticky=(N, W, E, S))

        capture_button = ttk.Button(
            self.__master,
            text="Capture",
            command=self.__capture
        )
        capture_button.grid(row=1, column=1, sticky=(N, W, E, S))

        self.__canvas = Canvas(
            self.__master, width=self.__image_size[0], height=self.__image_size[1])
        self.__canvas.grid(row=0, column=0, columnspan=2, sticky=(N, W, E, S))

    def __load_file(self) -> None:
        file_name = askopenfilename(
            filetypes=[("Image", "*.jpg;*.jpeg;*.png")])
        start_time = time.time()
        self.__predict(file_name)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution Time: {execution_time} seconds")

    def __capture(self) -> None:
        if platform.system() == "Windows":
            return
        subprocess.run([
            "gst-launch-1.0",
            "nvarguscamerasrc",
            "num-buffers=1",
            "!",
            "nvvidconv",
            "!",
            "video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=I420",
            "!",
            "nvjpegenc",
            "!",
            "filesink",
            f"location={self.__temp_path}"
        ])
        start_time = time.time()
        self.__predict(self.__temp_path)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution Time: {execution_time} seconds")

    def __predict(self, file_name: str) -> Tuple[Image.Image, float]:
        with Image.open(file_name) as image:
            image_np = np.array(image)
        output_image_np = self.__classifier.predict(image_np)
        output_image = Image.fromarray(output_image_np)
        output_image = output_image.resize(self.__image_size)

        self.__image_tk = ImageTk.PhotoImage(output_image)
        image_label = ttk.Label(self.__canvas, image=self.__image_tk)
        image_label.grid(row=0, column=0, sticky=(N, W, E, S))

    def run(self) -> None:
        self.__root.mainloop()
