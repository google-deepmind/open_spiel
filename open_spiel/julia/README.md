# Julia OpenSpiel

For general usage, please refer
[OpenSpiel on Julia](https://openspiel.readthedocs.io/en/latest/julia.html).

For developers, the basic idea of this Julia wrapper is that, a shared lib named
`libspieljl.so` is built with the help of
[CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl) and then it is wrapped
in the `OpenSpiel` module.
