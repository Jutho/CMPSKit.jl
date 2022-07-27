mutable struct FiniteCMPS{T<:MatrixFunction, N, V<:AbstractVector} <: LinearCMPS{T,N}
    Q::T
    Rs::NTuple{N,T}
    vL::V
    vR::V
    function FiniteCMPS(Q::T, Rs::NTuple{N,T}, vL::V, vR::V) where {T,N,V}
        for R in Rs
            domain(Q) == domain(R) || throw(DomainMismatch())
        end
        (a,b) = domain(Q)
        Qa = Q(a)
        length(vL) == length(vR) == size(Qa,1) == size(Qa,2) || throw(DimensionMismatch())
        for R in Rs
            size(R(a)) == size(Qa) || throw(DimensionMismatch())
        end
        return new{T,N,V}(Q, Rs, vL, vR)
    end
end
FiniteCMPS(Q::T, R::T, vL::V, vR::V) where {T,V} = FiniteCMPS(Q, (R,), vL, vR)

domain(Ψ::FiniteCMPS) = domain(Ψ.Q)

Base.iterate(Ψ::FiniteCMPS, args...) = iterate((Ψ.Q, Ψ.Rs, Ψ.vL, Ψ.vR), args...)

Base.copy(Ψ::FiniteCMPS) = FiniteCMPS(copy(Ψ.Q), copy.(Ψ.Rs), copy(Ψ.vL), copy(Ψ.vR))

Base.:(==)(Ψ1::FiniteCMPS, Ψ2::FiniteCMPS) =
    Ψ1.Q == Ψ2.Q && Ψ1.Rs == Ψ2.Rs && Ψ1.vL == Ψ2.vL && Ψ1.vR == Ψ1.vR

function LinearAlgebra.norm(Ψ::FiniteCMPS; kwargs...)
    Q, Rs, vL, vR = Ψ
    ρL, infoL = lefttransfer(vL*vL', nothing, Ψ; kwargs...)
    Z = vR'*ρL(b)*vR
    return sqrt(real(Z))
end

function LinearAlgebra.dot(Ψ₁::FiniteCMPS, Ψ₂::FiniteCMPS; kwargs...)
    a, b = domain(Ψ₁)
    (a, b) == domain(Ψ₂) || throw(DomainMismatch())
    Q₁, Rs₁, vL₁, vR₁ = Ψ₁
    Q₂, Rs₂, vL₂, vR₂ = Ψ₂
    ρL, infoL = lefttransfer(vL₂*vL₁', nothing, Ψ₁, Ψ₂; kwargs...)
    Z = vR₂'*ρL(b)*vR₁
    return Z
end

function LinearAlgebra.normalize!(Ψ::FiniteCMPS; kwargs...)
    ρL, λ, infoL = leftenv(Ψ; kwargs...)
    Ψ.Q -= λ * one(Ψ.Q)
    return Ψ
end
