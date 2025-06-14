from setuptools import setup, find_packages

setup(
    name='gymnasium_env',
    version='1.0.0',
    install_requires=[
        'geopy==2.4.1',
        'matplotlib==3.9.2',
        'networkx==3.3',
        'numpy==1.26.4',
        'osmnx==1.9.4',
        'pandas==2.2.3',
        'scipy==1.12.0',
        'tqdm==4.66.6',
        'sympy~=1.13.1',
        'geopandas~=0.14.4',
        'shapely~=2.0.6',
        'gymnasium~=1.0.0',
        'Cartopy~=0.24.1',
        'torch~=2.5.1',
        'torch-geometric~=2.6.1',
        'ipython~=8.29.0'
    ],
    packages=find_packages(),
    include_package_data=True,  # Include non-Python files if needed
)