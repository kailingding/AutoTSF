from setuptools import setup, find_packages
from codecs import open
from os import path

HERE = path.abspath(path.dirname(__file__))
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    README = f.read()
VERSION = '0.0.4'

# get the dependencies and installs
with open(path.join(HERE, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    name='Autotsf',
    description='Automated Time-Series Forecasting',
    url='https://github.com/kailingding/Autotsf',
    version=VERSION,
    license='MIT',
    packages=find_packages(exclude=("tests",)),
    long_description=README,
    long_description_content_type="text/markdown",
    include_package_data=True,
    author='Kailing Ding',
    install_requires=install_requires,
    setup_requires=['numpy>=1.10', 'scipy>=0.17'],
    dependency_links=dependency_links,
    author_email='markdingkl@gmail.com',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
)
