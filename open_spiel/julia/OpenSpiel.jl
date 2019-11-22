module Spiel

using CxxWrap

path_to_lib = get(ENV, "LIB_SPIEL_JL", "$(dirname(@__FILE__))/../../build/julia/libspieljl.so")
@wrapmodule(path_to_lib)

function __init__()
    @initcxx
end

end