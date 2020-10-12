from setuptools import setup, find_packages

setup(
    name='components',
    version='0.1',
    author='Santeri Rytky,Aleksei Tiulpin',
    author_email='santeri.rytky@oulu.fi',
    packages=find_packages(),
    include_package_data=True,
    license='LICENSE',
    long_description=open('README.md').read(),
)