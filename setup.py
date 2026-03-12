from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

import versioneer


long_description = Path(__file__).with_name("README.rst").read_text(encoding="utf-8")

setup(
    name="aotools",
    author_email="andrewpaulreeves@gmail.com, matthew.townson@durham.ac.uk",
    url="https://github.com/AOtools/aotools",
    packages=find_packages(exclude=("test", "doc")),
    description="A set of useful functions for Adaptive Optics in Python",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26.2",
        "scipy>=1.11.3",
        "matplotlib>=3.8",
        "numba>=0.59",
    ],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
