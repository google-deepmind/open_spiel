# GAMUT games

This is an interface to load normal-form games from the
[GAMUT](http://gamut.stanford.edu/) games generator. This interface is not
compiled with OpenSpiel by default and must be enabled via the
`OPEN_SPIEL_BUILD_WITH_GAMUT` environment variable (see the Developer Guide)
when OpenSpiel is built.

It requires a working JVM (`java` binary) and the `gamut.jar` from the GAMUT
project.

Note that this interface is not regularly tested, so it may break at any time.
Please open an issue to report any problem when using it.
