from setuptools import setup, find_packages

setup(
    name="geo_spde",
    version="0.5.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.6.0",
        "meshpy>=2020.1",
        "pyproj>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "examples": [
            "pandas>=1.3.0",
            "matplotlib>=3.3.0",
            "geopandas>=0.9.0",
        ]
    },
    python_requires=">=3.8",
    author="Mitzi Morris",
    description="SPDEs for geospatial data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
