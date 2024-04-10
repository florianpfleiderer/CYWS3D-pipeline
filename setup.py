from setuptools import setup, find_packages

setup(
    name='cyws3d-pipeline',
    version='0.2',
    description='A pipeline for 3D object annotation and visualization using cyws3d and the obchange dataset',
    author='Florian Pfleiderer',
    author_email='florian@pfleiderer.at',
    license='MIT',
    packages=find_packages(),
    scripts=['scripts/annotate.py']
    # requires=['numpy', 'open3d', 'opencv-python', 'matplotlib']
)
