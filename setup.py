from setuptools import setup, find_packages

setup(
    name="pusher_mastery",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium[mujoco]>=0.29.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
)