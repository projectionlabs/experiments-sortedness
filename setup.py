# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': 'src'}

packages = \
['experimentssortedness',
 'experimentssortedness.evaluation',
 'experimentssortedness.wtau']

package_data = \
{'': ['*']}

install_requires = \
['PyMySQL>=1.0.2,<2.0.0',
 'PyQt5>=5.15.7,<6.0.0',
 'cryptography>=37.0.4,<38.0.0',
 'dash>=2.6.0,<3.0.0',
 'hoshmap>=0.220808.0,<0.220809.0',
 'lazydf>=0.220725.4,<0.220726.0',
 'lz4>=4.0.2,<5.0.0',
 'matplotlib>=3.5.2,<4.0.0',
 'openml>=0.12.2,<0.13.0',
 'pandas>=1.4.3,<2.0.0',
 'pathos>=0.2.9,<0.3.0',
 'ranky>=0.2.9,<0.3.0',
 'rustil>=0.220918.1,<0.220919.0',
 'scikit-learn>=1.1.1,<2.0.0',
 'shelchemy>=0.220726.8,<0.220727.0',
 'sklearn>=0.0,<0.1',
 'sortedness>=0.220803.1,<0.220804.0',
 'sympy>=1.10.1,<2.0.0']

setup_kwargs = {
    'name': 'experimentssortedness',
    'version': '0.220803.5',
    'description': 'Comparison of measures of projection quality',
    'long_description': 'None',
    'author': 'davips',
    'author_email': 'dpsabc@gmail.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.10,<4.0',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
