# Generic implementation: arbitrary list of elements on the intervals between the nodes
struct Piecewise{T,F<:FunctionSpace{T},S<:AbstractVector{<:Real}} <: AbstractPiecewise{T,F}
    nodes::S
    elements::Vector{F}
    function Piecewise(nodes::S, elements::Vector{F}) where {T, F<:FunctionSpace{T}, S<:AbstractVector}
        @assert length(nodes) == length(elements) + 1
        @assert eltype(nodes) <: Real
        if nodes isa AbstractRange
            @assert step(nodes) > 0
        else
            @assert issorted(nodes)
        end
        return new{T, F, S}(nodes, elements)
    end
end

# Basic properties
nodes(p::Piecewise) = p.nodes
elements(p::Piecewise) = p.elements
nodevalues(p::Piecewise) = Base.Generator(p, nodes(p))

Base.length(p::Piecewise) = length(p.elements)

# Indexing, getting and setting elements
@inline Base.getindex(p::Piecewise, i) = getindex(p.elements, i)
@inline Base.setindex!(p::Piecewise, v, i) = setindex!(p.elements, v, i)

# Use as function
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

# Change number of coefficients
function truncate!(p::Piecewise; kwargs...)
    for i = 1:length(p)
        truncate!(p[i]; kwargs..., dx = (p.nodes[i+1]-p.nodes[i])/2)
    end
    return p
end

# Special purpose constructor
Base.similar(p::Piecewise, args...) =
    Piecewise(nodes(p), map(x->similar(x, args...), elements(p)))
Base.zero(p::Piecewise) = Piecewise(nodes(p), map(zero, elements(p)))
Base.one(p::Piecewise) = Piecewise(nodes(p), map(one, elements(p)))

# Arithmetic (out of place)
Base.copy(p::AbstractPiecewise) = Piecewise(copy(nodes(p)), map(copy, elements(p)))

Base.:+(p::AbstractPiecewise) = Piecewise(nodes(p), map(+, elements(p)))
Base.:-(p::AbstractPiecewise) = Piecewise(nodes(p), map(-, elements(p)))

function Base.:+(p1::AbstractPiecewise, p2::AbstractPiecewise)
    @assert nodes(p1) == nodes(p2)
    return Piecewise(nodes(p1), [p1[i] + p2[i] for i = 1:length(p1)])
end

function Base.:-(p1::AbstractPiecewise, p2::AbstractPiecewise)
    @assert nodes(p1) == nodes(p2)
    return Piecewise(nodes(p1), [p1[i] - p2[i] for i = 1:length(p1)])
end

Base.:*(p::AbstractPiecewise, a::Const) = Piecewise(nodes(p), map(x->x*a, elements(p)))
Base.:*(a::Const, p::AbstractPiecewise) = Piecewise(nodes(p), map(x->a*x, elements(p)))
Base.:/(p::AbstractPiecewise, a) = Piecewise(nodes(p), map(x->x/a, elements(p)))
Base.:\(a, p::AbstractPiecewise) = Piecewise(nodes(p), map(x->a\x, elements(p)))

Base.:*(p1::AbstractPiecewise, p2::AbstractPiecewise) = truncmul(p1, p2)
function truncmul(p1::AbstractPiecewise, p2::AbstractPiecewise; kwargs...)
    n = nodes(p1)
    @assert n == nodes(p2)
    return Piecewise(n, [truncmul(p1[i], p2[i]; kwargs..., dx = n[i+1]-n[i]) for i = 1:length(p1)])
end

for f in (:conj, :adjoint, :transpose, :real, :imag)
    @eval Base.$f(p::AbstractPiecewise) = Piecewise(nodes(p), map($f, elements(p)))
end

LinearAlgebra.tr(p::AbstractPiecewise) = Piecewise(nodes(p), map(tr, elements(p)))

# Arithmetic (in place / mutating methods)
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

function LinearAlgebra.mul!(pdst::Piecewise, α::Const, psrc::Piecewise)
    @assert nodes(pdst) == nodes(psrc)
    for i = 1:length(psrc)
        mul!(pdst[i], α, psrc[i])
    end
    return pdst
end

function LinearAlgebra.mul!(pdst::Piecewise, psrc::Piecewise, α::Const)
    @assert nodes(pdst) == nodes(psrc)
    for i = 1:length(psrc)
        mul!(pdst[i], psrc[i], α)
    end
    return pdst
end

function LinearAlgebra.axpy!(α, px::AbstractPiecewise, py::Piecewise)
    @assert nodes(px) == nodes(py)
    for i = 1:length(px)
        axpy!(α, px[i], py[i])
    end
    return py
end
function LinearAlgebra.axpby!(α, px::AbstractPiecewise, β, py::Piecewise)
    @assert nodes(px) == nodes(py)
    for i = 1:length(px)
        axpby!(α, px[i], β, py[i])
    end
    return py
end

function truncmul!(p::Piecewise, p1::AbstractPiecewise, p2::AbstractPiecewise,
                                α = true, β = false; kwargs...)
    @assert nodes(p) == nodes(p1) == nodes(p2)
    n = nodes(p)
    @inbounds for i = 1:length(p)
        truncmul!(p[i], p1[i], p2[i], α, β; kwargs..., dx = n[i+1]-n[i])
    end
    return p
end

function LinearAlgebra.mul!(p::Piecewise, p1::AbstractPiecewise, p2::AbstractPiecewise,
                                α = true, β = false)
    @assert nodes(p) == nodes(p1) == nodes(p2)
    @inbounds for i = 1:length(p)
        mul!(p[i], p1[i], p2[i], α, β)
    end
    return p
end

# Inner product and norm
function localdot(p1::AbstractPiecewise, p2::AbstractPiecewise)
    @assert nodes(p1) == nodes(p2)
    return Piecewise(nodes(p1), map(localdot, elements(p1), elements(p2)))
end

LinearAlgebra.dot(p1::AbstractPiecewise, p2::AbstractPiecewise) = integrate(localdot(p1, p2), domain(p1))

LinearAlgebra.norm(p::AbstractPiecewise) = sqrt(dot(p, p))

# Differentiate and integrate
differentiate(p::AbstractPiecewise) = Piecewise(nodes(p), map(differentiate, elements(p)))
function integrate(p::AbstractPiecewise, interval = domain(p))
    @assert interval == domain(p)
    s = integrate(p[1], (nodes(p)[1], nodes(p)[2]))
    for i = 2:length(p)
        s += integrate(p[i], (nodes(p)[i], nodes(p)[i+1]))
    end
    return s
end
