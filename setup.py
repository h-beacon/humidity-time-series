from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Predicting the earth humidity value using sequential NN models trained using measured data (LoRa end device under ground + yr.no api).',
    author='h-beacon',
    license='MIT',
)
