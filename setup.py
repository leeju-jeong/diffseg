from setuptools import setup, find_packages

setup(
    name='mmseg',
    version='0.16.0',
    description='OpenMMLab Semantic Segmentation Toolbox',
    author='OpenMMLab',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'mmcv>=1.3.7,<1.4.0',
    ],
)

###