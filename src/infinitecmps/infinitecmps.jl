const PeriodicMatrixFunction{T} = Union{Constant{<:AbstractMatrix{T}},
                                        FourierSeries{<:AbstractMatrix{T}}}

# Gauges:
# :n => no particular gauge, left and right fixed points completely generic
# :l => left gauge: identity left fixed point, generic right fixed point
# :L => left canonical form: identity left fixed point, diagonal right fixed point
# :r => right gauge
# :R => right canonical form:
# :s => symmetric: left and right fixed point identical
# :S => symmetric: left and right fixed point identical and diagonal

mutable struct InfiniteCMPS{T<:PeriodicMatrixFunction,N} <: AbstractCMPS{T,N}
    Q::T
    Rs::NTuple{N,T}
    gauge::Symbol
    function InfiniteCMPS(Q::T, Rs::NTuple{N,T}; gauge::Symbol = :n) where {T,N}
        for R in Rs
            domain(R) == domain(Q) || throw(DomainMismatch())
        end
        Q0 = Q[0]
        size(Q0, 1) == size(Q0, 2) || throw(DimensionMismatch())
        for R in Rs
            size(R[0]) == size(Q0) || throw(DimensionMismatch())
        end
        return new{T,N}(Q, Rs, gauge)
    end
end
InfiniteCMPS(Q::T, R::T; kwargs...) where T = InfiniteCMPS(Q, (R,); kwargs...)

domain(::InfiniteCMPS) = (-Inf, +Inf)
period(Ψ::InfiniteCMPS) = period(Ψ.Q)

scalartype(::InfiniteCMPS{T}) where T = scalartype(T)

Base.copy(Ψ::InfiniteCMPS) = InfiniteCMPS(copy(Ψ.Q), map(copy, Ψ.Rs); gauge = Ψ.gauge)

virtualdim(Ψ::InfiniteCMPS) = size(Ψ.Q[0], 1)

const UniformCMPS = InfiniteCMPS{<:Constant{<:AbstractMatrix}}
const FourierCMPS = InfiniteCMPS{<:FourierSeries{<:AbstractMatrix}}
