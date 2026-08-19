from setuptools import setup

setup(
    name="muscall",
    version="0.1.0",
    package_dir={
        "muscall": ".",
        "muscall.models": "models",
        "muscall.modules": "modules",
        "muscall.trainers": "trainers",
        "muscall.utils": "utils",
        "muscall.scripts": "scripts",
    },
    packages=[
        "muscall",
        "muscall.models",
        "muscall.modules",
        "muscall.trainers",
        "muscall.utils",
        "muscall.scripts",
    ],
)
