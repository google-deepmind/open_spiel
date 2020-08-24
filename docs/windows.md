# Windows Installation using Windows Subsystem for Linux (WSL)

## Purpose of this document

Defines the installation steps to get OpenSpiel running in a Windows 10
environment using WSL. Note that WSL does not include GPU support, so will run
on CPU only.

## Process

This process has been written for Windows 10, and tested on Windows 10 build
1903 (March 2019).

1.  Install the Windows Subsystem for Linux:

    Run the following command in Windows Powershell:

    ```powershell
    Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
    ```

2.  Install Ubuntu Linux from the Windows Store. Currently this is version
    18.04::

    Open up the Windows Store. Search for Ubuntu. Open up Ubuntu and press "Get"
    to install this.

3.  First time run of Ubuntu:

    Click on the Start Button and choose the Ubuntu icon. Wait until the distro
    installs. Provide a username and password for the default user account. Note
    that this account is a member of the Linux administrators (sudo) group so
    choose a secure username and password combination.

4.  Update / Upgrade packages (optional step)

    ```bash
    sudo apt-get update
    sudo apt-get upgrade
    ```

5.  Run through the first part of the OpenSpiel installation

    ```bash
    git clone https://github.com/deepmind/open_spiel.git
    cd open_spiel
    ./install.sh # you will be prompted for the password created at stage 3. Press Y to continue and install. During installation press Yes to restart services during package upgrades
    pip install -U pip # Upgrade pip (required for TF >= 1.15)
    pip3 install --upgrade -r requirements.txt # Install Python dependencies
    ```

6.  Now need to upgrade make version as the version of make which comes with
    Ubuntu 18.04 is not high enough to build OpenSpiel. (Note, this step won't
    be necessary if the version of Ubuntu in the Windows store gets upgraded to
    19.04)

    ```bash
    cd ..
    wget http://www.cmake.org/files/v3.12/cmake-3.12.4.tar.gz
    tar -xvzf cmake-3.12.4.tar.gz
    cd cmake-3.12.4/
    ./configure
    make
    sudo make install
    sudo update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1 --force
    cd ../open_spiel
    ```

7.  Finally, continue with the installation and run tests.

    ```bash
    mkdir build
    cd build
    CXX=g++ cmake -DPython3_EXECUTABLE=$(which python3) -DCMAKE_CXX_COMPILER=g++ ../open_spiel
    make -j12 # The 12 here is the number of parallel processes used to build
    ctest -j12 # Run the tests to verify that the installation succeeded
    ```

    The CMake variable `Python3_EXECUTABLE` is used to specify the Python
    interpreter. If the variable is not set, CMake's FindPython3 module will
    prefer the latest version installed. Note, Python >= 3.6.0 is required.

    One can run an example of a game running (in the `build/` folder):

    ```bash
    ./examples/example --game=tic_tac_toe
    ```

8.  Setting Your PYTHONPATH environment variable

    To be able to import the Python code (both the C++ binding `pyspiel` and the
    rest) from any location, you will need to add to your PYTHONPATH the root
    directory and the `open_spiel` directory.

    When using a virtualenv, the following should be added to
    `<virtualenv>/bin/activate`. For a system-wide install, ddd it in your
    `.bashrc` or `.profile`.

    ```bash
    # For the python modules in open_spiel.
    export PYTHONPATH=$PYTHONPATH:/<path_to_open_spiel>
    # For the Python bindings of Pyspiel
    export PYTHONPATH=$PYTHONPATH:/<path_to_open_spiel>/build/python
    ```

9.  Running the first example

    In the `build` directory, running `examples/example` will prints out a list
    of registered games and the usage. Now, let’s play game of Tic-Tac-Toe with
    uniform random players:

    ```bash
    examples/example --game=tic_tac_toe
    ```
