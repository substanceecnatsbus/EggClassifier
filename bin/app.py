import json
from egg_classifier import EggClassifier, ClassifierType, EggClassifierUI


def main():
    ui = initialize()
    ui.run()


def initialize() -> EggClassifierUI:
    with open("resources/config.json") as fin:
        config = json.load(fin)

    classifier_type = ClassifierType.HISTOGRAM
    if config["classifier_type"] == "mobilenetv2":
        classifier_type = ClassifierType.MOBILENETV2

    classifier = EggClassifier(
        config["number_of_rows"], config["number_of_columns"],
        config["offset_x_percent"], config["offset_y_percent"],
        config["radius"], config["colors"], config["font"],
        config["font_size"], classifier_type, config["model_path"],
        config["classes"], config["prediction_threshold"],
        config["image_size"]
    )

    ui = EggClassifierUI(classifier, config["image_size_ui"])
    return ui


if __name__ == "__main__":
    main()
