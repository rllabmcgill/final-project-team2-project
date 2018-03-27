from setuptools import setup, find_packages
import sys, os

# Don't import gym module here, since deps may not be installed
for package in find_packages():
    if '_gym_' in package:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), package))
from package_info import USERNAME, VERSION

setup(name='{}-{}'.format(USERNAME, 'gym-mallard-ducks'),
    version=VERSION,
    description='Optimal Exploitation Strategies for an Animal Population in a Markovian Environment',
    url='https://github.com/ppaquette/gym_mallard_ducks',
    author='Philip Paquette, Stephanie Larocque',
    author_email='pcpaquette@gmail.com',
    license='MIT License',
    packages=[package for package in find_packages() if package.startswith(USERNAME)],
    zip_safe=False,
    install_requires=[ 'gym>=0.10.0' ],
)
