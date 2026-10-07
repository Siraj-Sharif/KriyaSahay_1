from setuptools import setup, find_packages

setup(
    name="neurogrip",
    version="0.1.0",
    description="NeuroGrip PC-side Hand Gesture CV System",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
)
