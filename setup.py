from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path:str)->List[str]: 
     """This function will return a list of requirements

     Args:
         file_path (str): path/to/requirements.text

     Returns:
         List[str]: List of packages to be installed
     """
     
     requirements = []
     with open(file_path) as file:
          requirements  = file.readlines()
          requirements = [req.replace("\n","") for req in requirements]

          if '-e .' in requirements:
               requirements.remove('-e .')

          
     return requirements
     
     
setup(
     name='ml-project',
     version='0.0.1',
     author='abdel homi',
     author_email='del.homi10@gmail.com',
     packages=find_packages(),
     install_required = get_requirements('requirements.txt')
)