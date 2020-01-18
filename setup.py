from setuptools import setup, find_packages


version = '0.2.0'

with open("requirements.txt", "r") as f:
    requirements = f.read()

setup(
    name='timeseries',
    version=version,
    requirements=requirements,
    packages=find_packages(),
    url='',
    license='GPL-3',
    author='lopezco',
    description='Time series functionalities'
)
