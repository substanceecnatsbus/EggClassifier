import numpy as np
from typing import Tuple, List
import tensorflow as tf
from abc import ABC, abstractmethod

tf.autograph.set_verbosity(3)


class Classifier(ABC):

    @abstractmethod
    def predict(self, data: np.ndarray) -> List[int]:
        pass

    @property
    def model(self) -> tf.keras.Model:
        return self.__model

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
