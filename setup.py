import os
import setuptools


with open("README.md", "r") as f:
    long_description = f.read()
with open("requirements.txt") as f:
    install_requires = [pkg.strip() for pkg in f.readlines() if pkg.strip()]
scripts = [
    os.path.join("bin", fname) for fname in os.listdir("bin")]

setuptools.setup(
    name="MICE2_mocks",
    version="1.1",
    author="Jan Luca van den Busch",
    description="This Repository provides code to create galaxy mock catalogues based on MICE galaxy catalogues.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KiDS-WL/MICE2_mocks",
    packages=setuptools.find_packages(),
    include_package_data=True,
    scripts=scripts,
    install_requires=install_requires)
