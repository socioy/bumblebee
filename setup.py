import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

try:
    with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "An AI-powered Python package for natural, human-like mouse and keyboard control."

setup(
    name="bumblebee",
    version="1.0.0",
    description="An AI-powered Python package for natural, human-like mouse and keyboard control.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="socioy",
    author_email="bumblebee@socioy.com",
    url="https://github.com/socioy/bumblebee",
    packages=find_packages(include=["bumblebee", "bumblebee.*"]),
    include_package_data=True,
    install_requires=[
        "numpy>=2.2.4",  
        "torch>=2.6.0",
        "scipy>=1.15.2",
        "pynput>=1.8.1",
        "PyAutoGUI>=0.9.54",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6, <4",  # Adjust for more Python versions if necessary
)
