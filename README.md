# SSH:
    $ ssh zchen@contijoch-bayern.ucsd.edu
    $ ssh -o "UserKnownHostsFile=/dev/null" -o "StrictHostKeyChecking=no" zchen@contijoch-bayern.ucsd.edu 

# file:
    $ /media/ExtraDrive/workspaces

# Install:

    $ pipenv install git+https://github.com/dvigneault/dvpy.git#egg=dvpy
    $ or pip install git+https://github.com/zhennongchen/dvpy.git#egg=dvpy
    $ search "how to install python package from github

The only dependency *not* installed by default is tensorflow, as we do not want to specify the GPU or CPU version.  These can be installed as follows:

    $ pipenv install tensorflow # CPU
    $ pipenv install tensorflow-gpu # GPU
    $ use pip install tensorflow-gpu==1.10.0 to install tensorflow
    $ only the CUDA-9.0 version is compatible with tensorflow 1.10.0, so check the CUDA version first
    $ if not 9.0, use https://yangcha.github.io/CUDA90/ to install

    $ use keras 2.2.4 pip install keras==2.2.4
# Run tests:

## Clone the repository:

    $ git clone git@github.com:DVigneault/dvpy.git ~/Developer/repositories/dvpy
    $ cd ~/Developer/repositories/dvpy

## Set up a development environment:
    $ pipenv shell

## Install a local copy, plus the dependencies:

    $ pipenv install -e ./
    $ pipenv install tensorflow # or tensorflow-gpu, as needed
    $ pipenv install pytest

## Run the tests!

    $ pytest                                      # Run all the tests
    $ pytest ./test/test_find_duplicates.py       # Run a specific test
    $ pytest ./test/test_find_duplicates.py -v    # Be verbose
    $ pytest ./test/test_find_duplicates.py -v -s # Be super verbose

