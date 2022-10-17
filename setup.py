#!/usr/bin/env python

from setuptools import find_packages, setup

if __name__ == '__main__':
    setup(
        name='hat',
        description='HAT',
        include_package_data=True,
        packages=find_packages(exclude=('options', 'data', 'experiments', 'results', 'tb_logger', 'wandb')),
        setup_requires=['cython', 'numpy'],
        zip_safe=False)
