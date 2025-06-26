from setuptools import find_packages,setup
from typing import List

def get_reqs(file_path:str)->List[str]:
    reqs = []
    with open (file_path) as fp:
        reqs = fp.readlines()
        reqs = [req.replace('\n','') for req in reqs ]
    
    if '-e .' in reqs:
        reqs.remove('-e .')

    return reqs

setup(
    name='ml_proj',
    version='0.0.1',
    author='Shikhar',
    author_email='shikharx08@gmail.com',
    packages=find_packages(),
    install_requires = get_reqs('requirements.txt')
)