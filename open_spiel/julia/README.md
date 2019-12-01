# Julia OpenSpiel

For general usage, please refer [OpenSpiel on Julia](https://openspiel.readthedocs.io/en/latest/julia.html).

For developers, the basic idea of this Julia wrapper is that, a shared lib named `libspieljl.so` is built by using [CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl) and then it is wrapped in the `OpenSpiel_jll` module.

## Q&A

1. Why is this package named `OpenSpiel_jll` but not `OpenSpiel`?

    The reason is that we plan to use [BinaryBuilder](https://github.com/JuliaPackaging/BinaryBuilder.jl) for the building process once the dependencies and APIs are stable. So by convention, this package is named `OpenSpiel_jll`. Another package named `OpenSpiel` will be registered later.

1. What is `StdVector`?

    `StdVector` is introduced in [CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl) recently. It is a wrapper of `std::vector` in the C++ side. Since that it is a subtype of `AbstractVector`, most functions should just work out of the box.

1. I can't find the `xxx` function/type in the Julia wrapper/The program exits unexpectedly.

    Although most of the functions and types should be exported, only `State` and `Game` related APIs are well tested. So if you encounter any error, please do not hesitate to create an issue.