struct LeftTransfer{T,N}
    Q₁::T
    Q₂::T
    R₁s::NTuple{N,T}
    R₂s::NTuple{N,T}
end

struct RightTransfer{T,N}
    Q₁::T
    Q₂::T
    R₁s::NTuple{N,T}
    R₂s::NTuple{N,T}
end

LeftTransfer(ψ₁::CMPS, ψ₂::CMPS = ψ₁) where {CMPS<:InfiniteCMPS} =
    LeftTransfer(ψ₁.Q, ψ₂.Q, ψ₁.Rs, ψ₂.Rs)

RightTransfer(ψ₁::CMPS, ψ₂::CMPS = ψ₁) where {CMPS<:InfiniteCMPS} =
    RightTransfer(ψ₁.Q, ψ₂.Q, ψ₁.Rs, ψ₂.Rs)

scalartype(::Type{<:LeftTransfer{T}}) where T = scalartype(T)
scalartype(::Type{<:RightTransfer{T}}) where T = scalartype(T)

const UniformLeftTransfer = LeftTransfer{<:Constant}
const UniformRightTransfer = RightTransfer{<:Constant}

function (TL::LeftTransfer)(x; kwargs...)
    y = similar(x, promote_type(scalartype(x), scalartype(TL)))
    truncmul!(y, TL.Q₁', x; kwargs...)
    truncmul!(y, x, TL.Q₂, 1, 1; kwargs...)
    z = similar(y)
    for (R₁, R₂) in zip(TL.R₁s, TL.R₂s)
        mul!(z, R₁', x)
        truncmul!(y, z, R₂, 1, 1; kwargs...)
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
