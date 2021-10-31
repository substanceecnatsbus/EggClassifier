from setuptools import setup

with open("requirements.txt") as fin:
    DEPENDENCIES = [dependency.rstrip() for dependency in fin.readlines()]

print(DEPENDENCIES)

setup(
    name="EggClassifier",
    version='1.0',
    description="Classifer for images of fertile and infertile eggs.",
    packages=["egg_classifier"],
    install_requires=DEPENDENCIES
)
