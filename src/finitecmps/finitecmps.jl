mutable struct FiniteCMPS{T<:MatrixFunction, N, V<:AbstractVector} <: AbstractCMPS{T,N}
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
