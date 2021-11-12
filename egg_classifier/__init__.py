from enum import Enum, auto
from typing import Dict, Tuple
import numpy as np
import tensorflow as tf
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
                 classes: list[str], prediction_threshold: float,
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
