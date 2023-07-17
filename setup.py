from setuptools import setup, find_packages


VERSION = '0.0.1'
DESCRIPTION = 'Helper Library for PyTorch'
LONG_DESCRIPTION = 'Includes training, testing and various other helper functions for PyTorch for ease of use'

# Setting up
setup(
    name="PyTorch Beacon",
    version=VERSION,
    author="Lukelele (Luke Ye)",
    author_email="lukelele2001@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['numpy', 'torch', 'torchvision', 'tqdm'],
    keywords=[]
)