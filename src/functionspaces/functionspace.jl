function domain end
function differentiate end
function integrate end
function localdot end

const ∂ = differentiate
const ∫ = integrate

differentiate(a::Union{Number,AbstractArray}) = zero(a)
localdot(a::T, b::T) where {T<:Union{Number,AbstractArray}} = dot(a, b)

struct DomainMismatch <: Exception end

Base.show(io::IO, ::DomainMismatch) =
    Base.print(io, "DomainMismatch(): function space arguments have non-matching domain.")

abstract type FunctionSpace{T} end

scalartype(::Type{<:FunctionSpace{T}}) where T = scalartype(T)
Base.eltype(::Type{<:FunctionSpace{T}}) where T = T


function LinearAlgebra.isapprox(x::FunctionSpace, y::FunctionSpace;
                                atol::Real=0,
                                rtol::Real=Base.rtoldefault(eltype(x), eltype(y), atol))
    return norm(x-y) <= max(atol, rtol*max(norm(x), norm(y)))
end
