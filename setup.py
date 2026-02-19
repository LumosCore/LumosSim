from setuptools import find_packages
from skbuild import setup
# from setuptools import setup

setup(
    name='rapidAIsim',
    version='0.1',
    long_description=open('README.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'gurobipy==11.0.2',
        'ortools==9.10.4067',
        'numpy',
        'networkx',
        'pybind11',
        'mmh3',
        'scikit-build',
        'random2~=1.0.1',
        'py2opt',
        'jinja2',
        'pandas',
        'seaborn',
        'gym',
        'tqdm',
        'scikit-learn'
    ],
    packages=find_packages(exclude=['base_conf_template', 'large_exp_*']),
    package_dir={
        'rapidAIsim': 'rapidAIsim',
        # 'figret': 'figret'
    },
    package_data={
        'rapidAIsim': [
            'scheduler/static_locality_AI/leaf_spine_link_selector/saved_models/*.pth',
            'scheduler/static_locality_AI/leaf_spine_link_selector/saved_models/**/*.pth',
            'scheduler/ocsexpander/*.cpp',
            'scheduler/ocsexpander/*.h',
            'scheduler/ocsexpander/CMakeLists.txt'
        ]
    },
    entry_points={
        'console_scripts': [
            'rapidAIsim=rapidAIsim.main:main'
        ],
    },
    # cmake_install_dir='rapidAIsim',
    cmake_args=[
        '-DCMAKE_CXX_COMPILER=g++',
        '-DCMAKE_C_COMPILER=gcc',
        '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_CXX_FLAGS=-O3 -fPIC',
    ]
)
