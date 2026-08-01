from setuptools import setup, find_packages

setup(
    name="generative-stitch",
    version="0.1",
    author="Max Winokan",
    author_email="mwinokan@me.com",
    description="Create SVGs for eventual machine embroidery",
    url="https://github.com/mwinokan/generative-stitch",
    packages=find_packages(),
    python_requires="==3.12",
    install_requires=[
        "bezier",
    ],
)
