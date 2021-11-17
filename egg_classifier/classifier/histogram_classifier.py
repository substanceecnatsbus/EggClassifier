from typing import Tuple, Any, List
import os
import numpy as np
from PIL import Image
import tensorflow as tf

tf.autograph.set_verbosity(3)


class HistogramClassifier():

    def __init__(self, model_path: str, image_size: Tuple[int, int] = ...) -> None:
        self.__image_size = image_size
        self.__model = HistogramClassifier.create_model()
        self.__model.load_weights(model_path).expect_partial()

    @property
    def image_size(self):
        return self.__image_size

    @property
    def model(self) -> tf.keras.Model:
        return self.__model

    @staticmethod
    def load_dataset(dataset_path: str, image_size: Tuple,
                     batch_size: int = 16, test_split: float = 0.1,
                     seed: int = 123) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                               Tuple[np.ndarray, np.ndarray],
                                               List[str]]:
        if test_split >= 1:
            raise Exception("test_split must be less than 1")

        data = []
        labels = []
        fertile_path = f"{dataset_path}/fertile"
        for file_name in os.listdir(fertile_path):
            file = f"{fertile_path}/{file_name}"
            with Image.open(file) as image:
                resized_image = image.resize(image_size)
                histogram = resized_image.histogram()
            x = np.array(histogram, dtype=np.float)
            data.append(x)
            labels.append(0)
        infertile_path = f"{dataset_path}/infertile"
        for file_name in os.listdir(infertile_path):
            file = f"{infertile_path}/{file_name}"
            with Image.open(file) as image:
                resized_image = image.resize(image_size)
                histogram = resized_image.histogram()
            x = np.array(histogram, dtype=np.float)
            data.append(x)
            labels.append(1)

        data_count = len(data)
        data = np.array(data, dtype=np.float)
        labels = np.array(labels)

        idx = np.random.permutation(data_count)
        data, labels = data[idx], labels[idx]
        train_data_count = int(data_count * (1 - test_split))

        train_data = data[:train_data_count]
        train_labels = labels[:train_data_count]
        test_data = data[train_data_count:]
        test_labels = labels[train_data_count:]
        classes = ["fertile", "infertile"]

        return ((train_data, train_labels), (test_data, test_labels), classes)

    @staticmethod
    def create_model() -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(768))
        x = tf.keras.layers.LayerNormalization()(inputs)
        x = tf.keras.layers.Dense(
            16, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(0.02))(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        outputs = tf.keras.layers.Dense(
            1, activation=tf.keras.activations.sigmoid)(x)
        model = tf.keras.Model(inputs, outputs)
        return model

    @staticmethod
    def train(image_size: Tuple, dataset_path: str,
              save_path: str = "", number_of_epochs: int = 100,
              learning_rate: int = 0.0001, test_split: float = 0.1) -> Tuple[tf.keras.Model, Any]:

        model = HistogramClassifier.create_model()
        (train_data, train_labels), (test_data, test_labels), _ = HistogramClassifier.load_dataset(
            dataset_path, image_size, test_split=test_split)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        history = model.fit(x=train_data, y=train_labels,
                            epochs=number_of_epochs,
                            validation_data=(test_data, test_labels))

        if save_path != "":
            model.save_weights(f"{save_path}")
        return (model, history)

    def predict(self, data: np.ndarray) -> List[int]:
        number_of_images = data.shape[0]
        histograms = []
        for i in range(number_of_images):
            image = data[i, :]
            image = Image.fromarray(image)
            image = image.resize(self.image_size)
            histogram = image.histogram()
            histograms.append(histogram)
        inputs = np.array(histograms, dtype=np.float)
        predictions = self.model.predict(inputs)
        predictions = [1 if prediction >
                       0.4 else 0 for prediction in predictions]
        return predictions
