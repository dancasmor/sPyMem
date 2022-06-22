from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='sPyMem',
    version="0.0.3",
    author="Daniel Casanueva Morato",
    author_email="dcasanueva@us.es",
    description="An open-source package that offer fully-functional spike-based bio-inspired hippocampal memory models implemented with SNN technology in the SpiNNaker hardware.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dancasmor/sPyMem",
    project_urls={
        "Bug Tracker": "https://github.com/dancasmor/sPyMem/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.6",
    include_package_data=True,
)
