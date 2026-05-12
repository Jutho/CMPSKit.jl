# Transfer matrices: subscript 1 => ket matrices, subscript 2 => bra matrices

struct LeftTransfer{T<:MatrixFunction,N}
    Q₁::T
    R₁s::NTuple{N,T}
    Q₂::T
    R₂s::NTuple{N,T}
end

struct RightTransfer{T,N}
    Q₁::T
    R₁s::NTuple{N,T}
    Q₂::T
    R₂s::NTuple{N,T}
end
LeftTransfer(Q::T, Rs::NTuple{N,T}) where {T<:MatrixFunction,N} = LeftTransfer(Q, Rs, Q, Rs)
function RightTransfer(Q::T, Rs::NTuple{N,T}) where {T<:MatrixFunction,N}
    return RightTransfer(Q, Rs, Q, Rs)
end

function LeftTransfer(Ψ₁::CMPS, Ψ₂::CMPS=Ψ₁) where {CMPS<:AbstractCMPS}
    domain(Ψ₁) == domain(Ψ₂) || throw(DomainMismatch())
    return LeftTransfer(Ψ₁.Q, Ψ₁.Rs, Ψ₂.Q, Ψ₂.Rs)
end

function RightTransfer(Ψ₁::CMPS, Ψ₂::CMPS=Ψ₁) where {CMPS<:AbstractCMPS}
    domain(Ψ₁) == domain(Ψ₂) || throw(DomainMismatch())
    return RightTransfer(Ψ₁.Q, Ψ₁.Rs, Ψ₂.Q, Ψ₂.Rs)
end

scalartype(::Type{<:LeftTransfer{T}}) where {T} = scalartype(T)
scalartype(::Type{<:RightTransfer{T}}) where {T} = scalartype(T)

const UniformLeftTransfer = LeftTransfer{<:Constant}
const UniformRightTransfer = RightTransfer{<:Constant}

function (TL::LeftTransfer)(x; kwargs...)
    y = similar(x, promote_type(scalartype(x), scalartype(TL)))
    truncmul!(y, TL.Q₂', x; kwargs...)
    truncmul!(y, x, TL.Q₁, 1, 1; kwargs...)
    z = similar(y)
    for (R₁, R₂) in zip(TL.R₁s, TL.R₂s)
        mul!(z, R₂', x)
        truncmul!(y, z, R₁, 1, 1; kwargs...)
    end
    return y
end

function (TR::RightTransfer)(x; kwargs...)
    y = similar(x, promote_type(scalartype(x), scalartype(TR)))
    truncmul!(y, TR.Q₁, x; kwargs...)
    truncmul!(y, x, TR.Q₂', 1, 1; kwargs...)
    z = similar(y)
    for (R₁, R₂) in zip(TR.R₁s, TR.R₂s)
        mul!(z, R₁, x)
        truncmul!(y, z, R₂', 1, 1; kwargs...)
    end
    return y
end

function _full(𝕋::Union{LeftTransfer,RightTransfer}; kwargs...)
    Q₁ = 𝕋.Q₁
    R₁s = 𝕋.R₁s
    Q₂ = 𝕋.Q₂
    R₂s = 𝕋.R₂s
    T = map_bilinear(⊗, Q₁, one(Q₂))
    T = axpy!(1, map_bilinear(⊗, one(Q₁), conj(Q₂)), T)
    for (R₁, R₂) in zip(R₁s, R₂s)
        T = axpy!(1, map_bilinear(⊗, R₁, conj(R₂)), T)
    end
    return T
end
