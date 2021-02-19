import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="traditional_ml",
    version="0.0.0b",
    author="Md. Shamim Towhid",
    author_email="shamim.towhid@gmail.com",
    description="A traditional machine learning library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shamimtowhid/traditional_ML",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
