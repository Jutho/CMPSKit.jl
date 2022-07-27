abstract type AbstractMatrixFunctionSet{N,T} end

"""
    struct DiagonalBosonicMatrices{N,T}

Represents `N` commuting matrices by enforcing that they have a joint diagonal form
``R_k = V * E_k * inv(V)``. The struct stores `V::Matrix{T}`, `iV = inv(V)`, the diagonals `Es::NTuple{N,Vector{T}}` and the actual matrices `Rs::NTuple{N,Matrix{T}}`. Indexing returns `Rs[k]`. The same struct can store a tangent vector.
"""

struct DiagonalBosonicMatrices{N,T} < AbstractMatrixFunctionSet{N,Constant{Matrix{T}}}
    V::Matrix{T}
    iV::Matrix{T}
    Es::NTuple{N, Vector{T}}
    Rs::NTuple{N, Constant{Matrix{T}}}
end

function DiagonalBosonicMatrices(V::Matrix, Es::NTuple{N,Vector)
    iV = inv(V)
    Rs = Constant.( (V,) .* Diagonal.(Es) .* (iV,) )
    return DiagonalBosonicMatrices(V, iV, Es, Rs)
end

DiagonalBosonicMatrices(V::Matrix, Es::Vararg{Vector}) = DiagonalBosonicMatrices(V, Es)

Base.iterate(B::DiagonalBosonicMatrices, args...) = iterate(B.Rs, args...)

Base.eltype(::Type{DiagonalBosonicMatrices{N,T}}) where {N,T} = Matrix{T}

Base.getindex(B::DiagonalBosonicMatrices, k::Int) = B.Rs[k]

Base.length(::DiagonalBosonicMatrices{N}) where N = N

Base.Broadcast.broadcastable(B::DiagonalBosonicMatrices) = B.Rs

function tangent_inner(dB1::DiagonalBosonicMatrices, dB2::DiagonalBosonicMatrices)
    dV1, dV2 = dB1.V, dB2.V
    s = 2*real(dot(dV1, dV2))
    for (dE1,dE2) in zip(B1.Es, B2.Es)
        s += 2*real(dot(dE1, dE2))
    end
    return s
end

function tangent_project(B::DiagonalBosonicMatrices{N}, dRs::NTuple{N,Constant{<:Matrix}})
    V, iV, Rs = B.V, B.iV, B.Rs
    dEs = map(dRs) do dR
        diag(B.V' * dR[] * iV')
    end
    dV = zero(V)
    for n = 1:N
        dV .+= (dRs[n]*Rs[n]' - Rs[n]'*dRs[n])
    end
    dV = dV * iV'
    diV = - iV * dV * iV
    dX = dV * iV
    dRs = Ref(dX) .* Rs .- Rs .* Ref(dX) .+ Constant.(Ref(V) .* Diagonal.(dEs) .* Ref(iV))
    dB = DiagonalBosonicMatrices(dV, diV, dEs, dRs)
    return dB
end

function tangent_retract(B::DiagonalBosonicMatrices{N}, α, dB::DiagonalBosonicMatrices{N})
    V, iV, Es = B.V, B.iV, B.Es
    dV, dEs = dB.V, dB.Es
    dX = dB.V * B.iV

    V′ = exp(α * dX) * B.V
    iV′ = inv(V)
    Es′ = Es .+ α .* dEs
    Rs′ = Constant.(Ref(V′) .* Diagonal.(Es′) .* Ref(iV′))
    B′ = DiagonalBosonicMatrices(V′, iV′, Es′, Rs′)

    dV′ = dX * V′
    diV = - iV′ * dX
    dEs′ = dEs
    dRs′ = Ref(dX) .* Rs′ .- Rs′ .* Ref(dX) .+
            Constant.(Ref(V′) .* Diagonal.(dEs′) .* Ref(iV′))
    dB′ = DiagonalBosonicMatrices(dV′, diV′, dEs′, dRs′)
    return B′, dB′
end
