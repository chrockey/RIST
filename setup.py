"""Setup script for RIST package."""

from setuptools import setup, find_packages


setup(
    name="rist",
    packages=find_packages(include=["src*"]),
    include_package_data=True,
    version="1.0.0",
    description="Learning SO(3)-Invariant Semantic Correspondence via Local Shape Transform",
    author="Chunghyun Park, Seungwook Kim, Jaesik Park, Minsu Cho",
    url="https://github.com/chrockey/RIST",
    python_requires=">=3.8",
)
