from setuptools import setup

setup(name='aenet',
      version='0.1',
      author='Naoya Takahashi, Michael Gygli',
      packages=['aenet'],
      install_requires=['numpy', 'moviepy', 'theano', 'lasagne'],
      zip_safe=False)
