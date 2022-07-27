mutable struct CircularCMPS{T<:PeriodicMatrixFunction, N, S} <: AbstractCMPS{T,N}
    Q::T
    Rs::NTuple{N,T}
    P::S
    function CircularCMPS(Q::T, Rs::NTuple{N,T}, P::S) where {T,N,S}
        for R in Rs
            domain(Q) == domain(R) || throw(DomainMismatch())
        end
        period(Q) == 0 || period(Q) == P || throw(DomainMismatch())
        Qa = Q(0)
        for R in Rs
            size(R(0)) == size(Qa) || throw(DimensionMismatch())
        end
        return new{T,N,S}(Q, Rs, P)
    end
end
CircularCMPS(Q::T, R::T, P = 1) where {T<:Constant} = CircularCMPS(Q, (R,), P)
CircularCMPS(Q::T, Rs::NTuple{N,T}) where {N, T<:Constant} = CircularCMPS(Q, Rs, 1)
CircularCMPS(Q::T, R::T) where {T<:FourierSeries} = CircularCMPS(Q, (R,), period(Q))
CircularCMPS(Q::T, Rs::NTuple{N,T}) where {N, T<:FourierSeries} =
    CircularCMPS(Q, Rs, period(Q))

const UniformCircularCMPS{A, N} = CircularCMPS{Constant{A}, N}

domain(Ψ::CircularCMPS) = (zero(period(Ψ)), period(Ψ))
period(Ψ::CircularCMPS) = Ψ.P

Base.iterate(Ψ::CircularCMPS, args...) = iterate((Ψ.Q, Ψ.Rs), args...)

Base.copy(Ψ::CircularCMPS) = CircularCMPS(copy(Ψ.Q), copy.(Ψ.Rs), period(Ψ))

Base.:(==)(Ψ1::CircularCMPS, Ψ2::CircularCMPS) =
    Ψ1.Q == Ψ2.Q && Ψ1.Rs == Ψ2.Rs

function LinearAlgebra.norm(Ψ::CircularCMPS; kwargs...)
    E = environment(Ψ; kwargs...)
    Z = tr(E(0))
    return sqrt(real(Z))
end

function LinearAlgebra.dot(Ψ₁::CircularCMPS, Ψ₂::CircularCMPS; kwargs...)
    period(Ψ₁) == period(Ψ₂) || throw(DomainMismatch())
    E = environment(Ψ₂, Ψ₁; kwargs...)
    return tr(E(0))
end

function LinearAlgebra.normalize!(Ψ::CircularCMPS; kwargs...)
    Ψ.Q = axpy!(-log(norm(Ψ))/period(Ψ), one(Ψ.Q), Ψ.Q)
    return Ψ
end

function expval(ops::LocalOperator, Ψ::CircularCMPS, E = nothing; kwargs...)
    if isnothing(E)
        E = environment(Ψ; kwargs...)
    end
    EO = zero(E)
    for (c, op) in zip(coefficients(ops), operators(ops))
        addkronecker!(EO[], getindex.(_ketbrafactors(op, Ψ.Q, Ψ.Rs))..., c)
    end
    Z = tr(E(0))
    return tr(EO * E)/Z
end
