import re
from setuptools import find_packages, setup


def get_version():
    with open("torchTraining/version.py") as f:
        exec(compile(f.read(), "torchTraining/version.py", "exec"))
    return locals()["__version__"]


setup(
    name="torchTraining",
    version=get_version(),
    description="Engine of torchTraining projects",
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
    ],
    extras_require={},
)
