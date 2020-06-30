

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="social-behavior-analysis",
    version="0.0.3",
    author="Ben Devlin ",
    author_email="benjamin.devlin@duke.edu",
    description="Package to facilitate analysis of DLC social behavior output",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bendevlin18/social-behavior-analysis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)