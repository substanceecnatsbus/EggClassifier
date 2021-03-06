from typing import Tuple, Any, List
import numpy as np
from PIL import Image
import tensorflow as tf
from egg_classifier.classifier import Classifier

tf.autograph.set_verbosity(3)


class Mobilenetv2Classifier(Classifier):

    def __init__(self, model_path: str, image_size: Tuple[int, int] = (128, 128), ) -> None:
        super().__init__(image_size=image_size)
        self.__model = Mobilenetv2Classifier.create_model(image_size)
        self.__model.load_weights(model_path).expect_partial()

    @property
    def model(self) -> tf.keras.Model:
        return self.__model

    def predict(self, data: np.ndarray) -> List[int]:
        number_of_images = data.shape[0]
        images = []
        for i in range(number_of_images):
            image = data[i, :]
            image = Image.fromarray(image)
            image = image.resize(self.image_size)
            image = np.array(image)
            images.append(image)
        inputs = np.array(images)
        predictions = self.model.predict(inputs)
        predictions = [1 if prediction >
                       0.4 else 0 for prediction in predictions]
        return predictions

    @staticmethod
    def load_dataset(dataset_path: str, image_size: Tuple,
                     batch_size: int = 16, test_split: float = 0.1,
                     seed: int = 123) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
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

    @staticmethod
    def create_model(image_size: Tuple) -> Tuple[tf.keras.Model, Any]:
        base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            input_shape=(image_size[0], image_size[1], 3), include_top=False, weights='imagenet',
            input_tensor=None, pooling=None,
        )
        base_model.trainable = False

        inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(
            1, activation=tf.keras.activations.sigmoid)(x)
        model = tf.keras.Model(inputs, outputs)
        return model

    @staticmethod
    def train(image_size: Tuple, dataset_path: str,
              save_path: str = "", number_of_epochs: int = 100,
              learning_rate: int = 0.0001, test_split: float = 0.1) -> Tuple[tf.keras.Model, Any]:
        model = Mobilenetv2Classifier.create_model(image_size)

        train_dataset, test_dataset, _ = Mobilenetv2Classifier.load_dataset(
            dataset_path, image_size, test_split=test_split)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        history = model.fit(train_dataset,
                            epochs=number_of_epochs,
                            validation_data=test_dataset)

        if save_path != "":
            model.save_weights(save_path)
        return (model, history)
