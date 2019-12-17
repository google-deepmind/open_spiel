# Installation using `pip`

This covers installation using the Python package manager `pip`.

## Summary

1.  Install system packages and download some dependencies. Only needs to be
    run once.

    ```bash
    ./install.sh
    ```

2.  Install your Python dependencies, e.g. in Python 3 using
    [`virtualenv`](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/):

    ```bash
    virtualenv -p python3 venv
    source venv/bin/activate
    python3 -m pip install .
    ```

    Optionally, add `-e` to the last command to install in
    [editable mode](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs),
    which will allow you to skip this step if you edit any Python source files.
    If you edit any C++ files, you will have to rerun the install command.

    Use `deactivate` to quit the virtual environment.

3.  Run tests to check everything works:

    ```bash
    pip install nox
    nox -s tests
    ```

To make sure OpenSpiel works on the default configurations, we do use the
`python3` command and not `python` (which still defaults to Python 2 on modern
Linux versions).

# Running the first example

In the `build` directory, running `examples/example` will prints out a list of
registered games and the usage. Now, let’s play game of Tic-Tac-Toe with uniform
random players:

```bash
examples/example --game=tic_tac_toe
```
