# Copyright 2014-2024 Carnegie Mellon University

import setuptools

DESCRIPTION = open("README.md").read()

setuptools.setup(name="pyibl",
      version="5.1.6.dev1",
      description="A Python implementation of a subset of Instance Based Learning Theory",
      license="Free for research purposes",
      author="Dynamic Decision Making Laboratory of Carnegie Mellon University",
      author_email="dfm2@cmu.edu",
      url="http://pyibl.ddmlab.com/",
      platforms=["any"],
      long_description=DESCRIPTION,
      long_description_content_type="text/markdown",
      py_modules=["pyibl"],
      packages=setuptools.find_packages(),
      tests_require=["pytest"],
      python_requires=">=3.8",
      classifiers=["Intended Audience :: Science/Research",
                   "License :: OSI Approved :: MIT License",
                   "Programming Language :: Python",
                   "Programming Language :: Python :: 3 :: Only",
                   "Programming Language :: Python :: 3.8",
                   "Programming Language :: Python :: 3.9",
                   "Programming Language :: Python :: 3.10",
                   "Programming Language :: Python :: 3.11",
                   "Operating System :: OS Independent"])
