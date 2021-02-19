import re
import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

VERSIONFILE = "traditional_ml/version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setuptools.setup(
    name="traditional_ml",
    version=verstr,
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
