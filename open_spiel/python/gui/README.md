# OpenSpiel GUI server

> **Note:** The OpenSpiel GUI is intended for personal research use on your
> own machine. It is not designed or reviewed as a production or secured
> system: do not expose it to untrusted networks or users. The server has no
> authentication, and the oracle viewer sees every player's private
> information.

A local web server that lets people play any OpenSpiel game in a browser. The
browser client lives in `open_spiel/gui/`.

Run it with `python3 -m open_spiel.python.gui`; `--help` lists its flags. It
listens on `127.0.0.1` by default. `CONTRACT.md` specifies its JSON API.

The tests are the `*_test.py` files beside the modules. They sweep a few
representative games; `python3 -m open_spiel.python.gui.sweep_all` sweeps
every registered game against `sweep_exceptions.json`, which takes minutes.
Run it after changing the server or adding a game.

The format, lint and type-check configuration is in `pyproject.toml`, which
also pins the tools that run it.
