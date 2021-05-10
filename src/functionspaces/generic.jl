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

# FunctionSeries
abstract type FunctionSpace{T} end # functions that support taking linear combinations
abstract type FunctionSeries{T} <: FunctionSpace{T} end
const Const = Union{Number, AbstractArray}

scalartype(::Type{<:FunctionSpace{T}}) where T = scalartype(T)
Base.eltype(::Type{<:FunctionSpace{T}}) where T = T

_rtoldefault(x, y, atol) =
    Base.rtoldefault(scalartype(eltype(x)), scalartype(eltype(y)), atol)

function LinearAlgebra.isapprox(x::F, y::F;
                                atol::Real=0,
                                rtol::Real=Base.rtoldefault(eltype(x), eltype(y), atol)
                                ) where {F<:FunctionSeries}
    return norm(x-y) <= max(atol, rtol*max(norm(x), norm(y)))
end

# AbstractPiecewise
abstract type AbstractPiecewise{T,F<:FunctionSpace{T}} <: FunctionSpace{T} end

domain(p::AbstractPiecewise) = (first(nodes(p)), last(nodes(p)))
function domain(p::AbstractPiecewise, i)
    1 <= i <= length(p) || throw(BoundsError(p, i))
    n = nodes(p)
    @inbounds begin
        return (n[i], n[i+1])
    end
end

Base.eltype(t::AbstractPiecewise{T}) where {T} = eltype(T)
Base.eltype(::Type{<:AbstractPiecewise{T}}) where {T} = eltype(T)
