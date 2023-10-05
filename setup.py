import re
from setuptools import find_packages, setup


def get_version():
    with open("TorchTrainer/version.py") as f:
        exec(compile(f.read(), "TorchTrainer/version.py", "exec"))
    return locals()["__version__"]


setup(
    name="TorchTrainer",
    version=get_version(),
    description="Engine of TorchTrainer projects",
    url="",
    author="",
    author_email="",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pyyaml",
        "torch",
        "torchvision",
        "matplotlib",
        "lmdb",
        "addict",
        "yapf",
    ],
    extras_require={},
)
