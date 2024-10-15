# setup.py

from setuptools import setup, find_packages

setup(
    name='shadowjourney',
    version='0.1.4',
    description='Client library for ShadowJourney API',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Ichate',
    author_email='ichate_@outlook.com',
    url='https://github.com/shadowjourneyy/shadowjourney',
    packages=find_packages(),
    install_requires=[
        'requests',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
