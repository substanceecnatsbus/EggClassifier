import numpy as np
from typing import Tuple, List
import tensorflow as tf
from abc import ABC, abstractmethod

tf.autograph.set_verbosity(3)


class Classifier(ABC):

    def __init__(self, image_size: Tuple[int, int] = ...) -> None:
        self.__image_size = image_size

    @property
    def image_size(self):
        return self.__image_size

    @property
    @abstractmethod
    def model(self) -> tf.keras.Model:
        pass

    @abstractmethod
    def predict(self, data: np.ndarray) -> List[int]:
        pass

    @staticmethod
    @abstractmethod
    def load_dataset(dataset_path: str, image_size: Tuple,
                     batch_size: int = 16, test_split: float = 0.1,
                     seed: int = 123) -> None:
        pass

    @staticmethod
    @abstractmethod
    def train(image_size: Tuple, dataset_path: str,
              save_path: str = "", number_of_epochs: int = 100,
              learning_rate: int = 0.0001, test_split: float = 0.1) -> None:
        pass
