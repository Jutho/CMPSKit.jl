function domain end
function differentiate end
function integrate end
function localdot end

const ∂ = differentiate
const ∫ = integrate

differentiate(a::Union{Number,AbstractArray}) = zero(a)
localdot(a::T, b::T) where {T<:Union{Number,AbstractArray}} = dot(a, b)

struct DomainMismatch <: Exception end

function Base.show(io::IO, ::DomainMismatch)
    return Base.print(io,
                      "DomainMismatch(): function space arguments have non-matching domain.")
end

# FunctionSeries
abstract type FunctionSpace{T} end # functions that support taking linear combinations
abstract type FunctionSeries{T} <: FunctionSpace{T} end
const Const = Union{Number,AbstractArray}

scalartype(::Type{<:FunctionSpace{T}}) where {T} = scalartype(T)
Base.eltype(::Type{<:FunctionSpace{T}}) where {T} = T

function _rtoldefault(x, y, atol)
    return Base.rtoldefault(scalartype(eltype(x)), scalartype(eltype(y)), atol)
end

function LinearAlgebra.isapprox(x::FunctionSpace, y::FunctionSpace;
                                atol::Real=0,
                                rtol::Real=_rtoldefault(eltype(x), eltype(y), atol))
    return norm(x - y) <= max(atol, rtol * max(norm(x), norm(y)))
end

LinearAlgebra.normalize!(f::FunctionSpace) = rmul!(f, 1 / norm(f))
LinearAlgebra.normalize(f::FunctionSpace) = f / norm(f)

# AbstractPiecewise
abstract type AbstractPiecewise{T,F<:FunctionSpace{T}} <: FunctionSpace{T} end

domain(p::AbstractPiecewise) = (first(nodes(p)), last(nodes(p)))
function domain(p::AbstractPiecewise, i)
    1 <= i <= length(p) || throw(BoundsError(p, i))
    n = nodes(p)
    @inbounds begin
        return (n[i], n[i + 1])
    end
end

Base.eltype(t::AbstractPiecewise{T}) where {T} = eltype(T)
Base.eltype(::Type{<:AbstractPiecewise{T}}) where {T} = eltype(T)

# Some general functionality
Base.copy(f::FunctionSpace) = map_linear(copy, f)
Base.:+(f::FunctionSpace) = map_linear(+, f)
Base.:-(f::FunctionSpace) = map_linear(-, f)
Base.conj(f::FunctionSpace) = map_antilinear(conj, f)
Base.transpose(f::FunctionSpace) = map_linear(transpose, f)
Base.adjoint(f::FunctionSpace) = map_antilinear(adjoint, f)
LinearAlgebra.tr(f::FunctionSpace) = map_linear(LinearAlgebra.tr, f)
localdot(f₁::FunctionSpace, f₂::FunctionSpace) = map_sesquilinear(dot, f₁, f₂)
