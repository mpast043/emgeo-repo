"""
emgeo: Emergent Geometry Toolkit
Setup configuration
"""

from setuptools import setup, find_packages

# Read long description from README
try:
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Computational framework for extracting spacetime from quantum substrates"

setup(
    name='emgeo',
    version='0.1.0',
    author='M. A. Turner',
    author_email='',  # Add if desired
    description='Extract emergent spacetime geometries from quantum substrates',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/maturner/emgeo',  # Update when repo created
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'matplotlib>=3.3.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov',
            'black',
            'flake8',
            'mypy',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme',
            'numpydoc',
        ],
        'examples': [
            'jupyter',
            'notebook',
        ],
    },
    keywords='quantum gravity emergent spacetime quantum field theory',
    project_urls={
        'Documentation': 'https://emgeo.readthedocs.io',  # When ready
        'Source': 'https://github.com/maturner/emgeo',
        'Paper': 'https://arxiv.org/abs/XXXX.XXXXX',  # Add arXiv number
    },
)
