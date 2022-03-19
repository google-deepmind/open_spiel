# Open Spiel Python API

This is a Python API for OpenSpiel.

See `python/examples/example.py` for an example use and overview of the API, and
the main OpenSpiel installation instructions to see how to run this example.

For the full API specification, please see `python/pybind11/pyspiel.cc`.

# Useful commands

```
export PYTHONPATH=$PYTHONPATH:/home/jakob/open_spiel
export PYTHONPATH=$PYTHONPATH:/home/jakob/open_spiel/build/python
./open_spiel/scripts/build_and_run_tests.sh --virtualenv=false 
cd open_spiel/open_spiel; ./scripts/regenerate_playthroughs.sh
cd ..; ./open_spiel/scripts/build_and_run_tests.sh --virtualenv=false
```