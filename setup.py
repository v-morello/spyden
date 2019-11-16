import os
from setuptools import setup, find_packages

MODULE_NAME = 'spyden'

install_requires = [
    'numpy>=1.13',
    'matplotlib>=2.0'
]


with open('README.md', 'r') as fh:
    long_description = fh.read()


def parse_version():
    """ Parse version number from designed unique location """
    thisdir = os.path.dirname(__file__)
    version_file = os.path.join(thisdir, MODULE_NAME, '_version.py')
    with open(version_file, 'r') as fobj:
        text = fobj.read()
    items = {}
    exec(text, None, items)
    return items['__version__']


setup(
    name=MODULE_NAME,
    url='https://bitbucket.org/vmorello/{}'.format(MODULE_NAME),
    author='Vincent Morello',
    author_email='vmorello@gmail.com',
    description='Functions to evaluate the signal-to-noise ratio of pulsar data in a mathematically correct fashion',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version=parse_version(),
    packages=find_packages(),
    install_requires=install_requires,
    license='MIT License',

    # NOTE (IMPORTANT): This means that everything mentioned in MANIFEST.in will be copied at install time 
    # to the package’s folder placed in 'site-packages'
    include_package_data=True,

    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Topic :: Scientific/Engineering :: Astronomy"
        ],
)