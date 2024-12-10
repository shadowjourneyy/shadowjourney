from setuptools import setup, find_packages

setup(
    name='shadowjourney',
    version='0.2.0',
    description='Python library for ShadowJourney API',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Ichate',
    author_email='ichate_@outlook.com',
    maintainer = 'Bananalolok',
    maintainer_email = "katz@katz.is-a.dev",
    url='https://github.com/shadowjourneyy/shadowjourney',
    packages=find_packages(),
    install_requires=[
        'requests',
        'aiohttp'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)