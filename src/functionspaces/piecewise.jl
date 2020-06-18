abstract type AbstractPiecewise{T} end

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

# Generic implementation: arbitrary list of elements on the intervals between the nodes
struct Piecewise{T,S<:AbstractVector{<:Real}} <: AbstractPiecewise{T}
    nodes::S
    elements::Vector{T}
    function Piecewise(nodes::S, elements::Vector{T}) where {S<:AbstractVector,T}
        @assert length(nodes) == length(elements) + 1
        @assert eltype(nodes) <: Real
        if nodes isa AbstractRange
            @assert step(nodes) > 0
        else
            @assert issorted(nodes)
        end
        return new{T, S}(nodes, elements)
    end
end

nodes(p::Piecewise) = p.nodes
elements(p::Piecewise) = p.elements
nodevalues(p::Piecewise) = Base.Generator(p, nodes(p))

Base.length(p::Piecewise) = length(p.elements)

@inline Base.getindex(p::Piecewise, i) = getindex(p.elements, i)
@inline Base.setindex!(p::Piecewise, v, i) = setindex!(p.elements, v, i)

function (p::Piecewise)(x)
    x₀ = first(nodes(p))
    x₁ = last(nodes(p))
    (x < x₀ || x > x₁) && throw(DomainError())
    x == x₀ && return p[1](x)
    x == x₁ && return p[length(p)](x)
    if nodes isa AbstractRange
        ir = (x-x₀)/step(nodes(p))
        i = floor(Int, ir)
        return p[i](x)
    else
        j = findlast(<=(x), nodes(p))
        @assert j !== nothing && j !== length(nodes(p))
        return p[j](x)
    end
end

Base.copy(p::Piecewise) = Piecewise(copy(nodes(p)), map(copy, elements(p)))

Base.similar(p::Piecewise) = Piecewise(nodes(p), map(similar, elements(p)))

function LinearAlgebra.rmul!(p::Piecewise, α)
    for i = 1:length(p)
        rmul!(p[i], α)
    end
    return p
end

function LinearAlgebra.lmul!(α, p::Piecewise)
    for i = 1:length(p)
        lmul!(α, p[i])
    end
    return p
end

function LinearAlgebra.mul!(pdst::Piecewise, α, psrc::Piecewise)
    @assert nodes(pdst) == nodes(psrc)
    for i = 1:length(psrc)
        mul!(pdst[i], α, psrc[i])
    end
    return pdst
end

function LinearAlgebra.mul!(pdst::Piecewise, psrc::Piecewise, α)
    @assert nodes(pdst) == nodes(psrc)
    for i = 1:length(psrc)
        mul!(pdst[i], psrc[i], α)
    end
    return pdst
end

function LinearAlgebra.axpy!(α, px::Piecewise, py::Piecewise)
    @assert nodes(px) == nodes(py)
    for i = 1:length(px)
        axpy!(α, px[i], py[i])
    end
    return py
end
function LinearAlgebra.axpby!(α, px::Piecewise, β, py::Piecewise)
    @assert nodes(px) == nodes(py)
    for i = 1:length(px)
        axpby!(α, px[i], β, py[i])
    end
    return py
end

Base.zero(p::Piecewise) = Piecewise(nodes(p), zero.(p.elements))
Base.one(p::Piecewise) = Piecewise(nodes(p), one.(p.elements))

function Base.:+(p1::AbstractPiecewise, p2::AbstractPiecewise)
    @assert nodes(p1) == nodes(p2)
    return Piecewise(nodes(p1), [p1[i] + p2[i] for i = 1:length(p1)])
end

function Base.:-(p1::AbstractPiecewise, p2::AbstractPiecewise)
    @assert nodes(p1) == nodes(p2)
    return Piecewise(nodes(p1), [p1[i] - p2[i] for i = 1:length(p1)])
end

Base.:-(p::AbstractPiecewise) = Piecewise(nodes(p), map(-, elements(p)))

Base.:*(p::AbstractPiecewise, a) = mul!(similar(p), p, a)
Base.:*(a, p::AbstractPiecewise) = mul!(similar(p), a, p)
Base.:/(p::AbstractPiecewise, a) = mul!(similar(p), p, inv(a))
Base.:\(a, p::AbstractPiecewise) = mul!(similar(p), inv(a), p)

for f in (:conj, :adjoint, :transpose, :real, :imag)
    @eval Base.$f(p::AbstractPiecewise) = Piecewise(nodes(p), map($f, elements(p)))
end

function Base.:*(p1::AbstractPiecewise, p2::AbstractPiecewise; kwargs...)
    @assert nodes(p1) == nodes(p2)
    return Piecewise(nodes(p1), [*(p1[i], p2[i]; kwargs...) for i = 1:length(p1)])
end

function LinearAlgebra.mul!(p::Piecewise, p1::AbstractPiecewise, p2::AbstractPiecewise,
                                α = true, β = false)
    @assert nodes(p) == nodes(p1) == nodes(p2)
    for i = 1:length(p)
        mul!(p[i], p1[i], p2[i], α, β)
    end
    return p
end

differentiate(p::AbstractPiecewise) = Piecewise(nodes(p), map(differentiate, elements(p)))
function integrate(p::AbstractPiecewise, interval = domain(p))
    @assert interval == domain(p)
    s = integrate(p[1], (nodes(p)[1], nodes(p)[2]))
    for i = 2:length(p)
        s += integrate(p[i], (nodes(p)[i], nodes(p)[i+1]))
    end
    return s
end

function localdot(p1::AbstractPiecewise, p2::AbstractPiecewise)
    @assert nodes(p1) == nodes(p2)
    return Piecewise(nodes(p1), [localdot(p1[i], p2[i]) for i = 1:length(p1)])
end

LinearAlgebra.tr(p::AbstractPiecewise) = Piecewise(nodes(p), map(tr, elements(p)))
