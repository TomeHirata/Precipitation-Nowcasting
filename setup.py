import os
import setuptools


setuptools.setup(
    name='nowcasting',
    version="0.1.dev0",
    packages=setuptools.find_packages(),
    license='MIT',
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas', 'moviepy', 'numba',
                      'pillow', 'six', 'easydict', 'pyyaml'],
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)
