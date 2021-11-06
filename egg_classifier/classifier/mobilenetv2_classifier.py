from typing import Tuple, Any
import tensorflow as tf
from egg_classifier.classifier import Classifier

tf.autograph.set_verbosity(3)


class Mobilenetv2Classifier(Classifier):

    @staticmethod
    def load_dataset(dataset_path: str, image_size: Tuple,
                     batch_size: int = 16, test_split: float = 0.1,
                     seed: int = 123) -> Tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
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
    def train(image_size: Tuple, dataset_path: str,
              save_path: str = "", number_of_epochs: int = 100,
              learning_rate: int = 0.0001, test_split: float = 0.1) -> Tuple[tf.keras.Model, Any]:
        base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            input_shape=image_size, include_top=False, weights='imagenet',
            input_tensor=None, pooling=None,
        )
        base_model.trainable = False

        inputs = tf.keras.Input(shape=image_size)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(
            1, activation=tf.keras.activations.sigmoid)(x)
        model = tf.keras.Model(inputs, outputs)

        train_dataset, test_dataset, _ = Mobilenetv2Classifier.load_dataset(
            dataset_path, (image_size[0], image_size[1]), test_split=test_split)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        history = model.fit(train_dataset,
                            epochs=number_of_epochs,
                            validation_data=test_dataset)

        if save_path != "":
            model.save(save_path)
        return (model, history)
