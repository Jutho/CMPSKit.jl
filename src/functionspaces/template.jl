# Type definition
struct SpecificFunctionSpaceType{T,...} <: FunctionSpace{T}
    # or some subtype FunctionSeries{T}, AbstractPiecewise{T}, ...
end

# Basic properties
domain
period
coefficients
...
Base.:(==)

# Indexing, getting and setting coefficients
Base.eachindex
Base.getindex
Base.setindex!
...

# Use as function
(f::SpecificFunctionSpaceType)(x)

# Change number of coefficients
truncate!

# Special purpose constructor
Base.similar
Base.zero
Base.one

# Arithmetic (out of place)
Base.copy
Base.:+  # unary and binary
Base.:- # unary and binary
Base.:* # multiplication with scalars
Base.:/ # with scalars
Base.:\ # with scalars
Base.:*(f1::FunctionSpaceType, f2::FunctionSpaceType)
truncmul

# Arithmetic (in place / mutating methods)
Base.copy!
LinearAlgebra.rmul!
LinearAlgebra.lmul!
LinearAlgebra.axpy!
LinearAlgebra.axpby!
LinearAlgebra.mul!
LinearAlgebra.truncadd!
LinearAlgebra.truncmul!

# Inner product and norm
LinearAlgebra.dot
LinearAlgebra.norm

# Differentiate and integrate
differentiate
integrate

# Apply linear and bilinear maps locally
map_linear
map_antilinear
map_bilinear
map_sesquilinear

Base.real # special case
Base.imag

# Inverse, square root (optional)
Base.inv
Base.sqrt

# Fit (optional)
fit
