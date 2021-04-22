from setuptools import setup

__version__ = "0.0.0"

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="chemetho",
    version=__version__,
    author="Tae Han (Han) Kim",
    author_email="hankim@caltech.edu",
    url="https://github.com/hanhanhan-kim/chemetho",
    description="Utilities for anazlying chemical ecology and ethology data",
    long_description=long_description,
    long_description_content_type="text/markdown", 

    license="GNU GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
    ],
    install_requires=[
        "pyyaml",
        "lxml", # for GC data
        "pandas", 
        "numpy", 
        "scipy", 
        "pyteomics", # for GC data
        "peakutils", # for GC data
        "bokeh", 
        "colorcet", 
        "cmocean",
        "iqplot"]
) 