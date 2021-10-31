import imp
from typing import Tuple
import tensorflow as tf
from tensorflow import keras


class Classifier:
    def __init__(self) -> None:
        pass

    @staticmethod
    def train():
        pass

    @staticmethod
    def load_dataset(dataset_path: str, image_size: Tuple[int, int], batch_size: int = 16, test_split: float = 0.1, seed: int = 123) -> Tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
        if test_split >= 1:
            raise Exception("test_split must be less than 1")

        train_dataset: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            subset="training",
            validation_split=test_split,
            image_size=image_size,
            batch_size=batch_size,
            seed=seed
        )
        test_dataset:  tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            subset="validation",
            validation_split=test_split,
            image_size=image_size,
            batch_size=batch_size,
            seed=seed
        )
        class_names = train_dataset.class_names

        return train_dataset, test_dataset, class_names
