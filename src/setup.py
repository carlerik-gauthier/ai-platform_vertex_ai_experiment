from setuptools import find_packages, setup
import os

path = os.path.abspath(__file__)
dir_name = os.path.dirname(path)
# with open(os.path.join(dir_name, 'requirements.txt')) as f:
with open('requirements.txt') as f:
    REQUIREMENTS = f.read()

# with open(os.path.join(dir_name, 'version')) as version_file:
with open('version') as version_file:
    VERSION = version_file.read().strip()

setup(
    name='ai-platform-tuto',
    version=VERSION,
    install_requires=REQUIREMENTS,
    packages=find_packages()
)
