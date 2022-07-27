⊗(A::AbstractMatrix, B::AbstractMatrix) = kron(B, A)

# Computes O = O + α * A ⊗ conj(B)
function addkronecker!(O::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α = one(eltype(O)))
    mA, nA = size(A)
    mB, nB = size(B)
    @assert size(O) == (mA*mB, nA*nB)
    O2 = reshape(O, (mA, mB, nA, nB))
    @inbounds for l = 1:nB
        for k = 1:nA
            for j = 1:mB
                @simd for i = 1:mA
                    O2[i,j,k,l] += α * A[i,k] * conj(B[j,l])
                end
            end
        end
    end
    return O
end
#
# function addkronecker2!(O::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α = one(eltype(O)))
#     mA, nA = size(A)
#     mB, nB = size(B)
#     @assert size(O) == (mA*mB, nA*nB)
#     O2 = reshape(O, (mA, mB, nA, nB))
#     @tturbo for l = 1:nB
#         for k = 1:nA
#             for j = 1:mB
#                 for i = 1:mA
#                     O2[i,j,k,l] += α * A[i,k] * conj(B[j,l])
#                 end
#             end
#         end
#     end
#     return O
# end

function addkronecker!(O::AbstractMatrix, A::AbstractMatrix, B::UniformScaling, α = one(eltype(O)))
    mA, nA = size(A)
    mB, rmB = divrem(size(O, 1), mA)
    nB, rnB = divrem(size(O, 2), nA)
    @assert mB == nB
    @assert rmB == rnB == 0
    O2 = reshape(O, (mA, mB, nA, nB))
    α2 = α * conj(B[1,1])
    @inbounds for k = 1:nA
        for j = 1:mB
            @simd for i = 1:mA
                O2[i,j,k,j] += α2 * A[i,k]
            end
        end
    end
    return O
end

function addkronecker!(O::AbstractMatrix, A::UniformScaling, B::AbstractMatrix, α = one(eltype(O)))
    mB, nB = size(B)
    mA, rmA = divrem(size(O, 1), mB)
    nA, rnA = divrem(size(O, 2), nB)
    @assert mA == nA
    @assert rmA == rnA == 0
    O2 = reshape(O, (mA, mB, nA, nB))
    α2 = α * A[1,1]
    @inbounds for l = 1:nB
        for i = 1:mA
            @simd for j = 1:mB
                O2[i,j,i,l] += α2 * conj(B[j,l])
            end
        end
    end
    return O
end
