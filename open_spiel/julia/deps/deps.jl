const LIB_OPEN_SPIEL = begin
    lib_path = "$(dirname(@__FILE__))/../../../build/julia/libspieljl.so"
    if !isfile(lib_path)
        @error "libspieljl.so not found. Please check the build status!"
    end
    lib_path
end