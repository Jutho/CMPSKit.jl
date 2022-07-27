function partialtrace1(A::AbstractMatrix, D1, D2)
    size(A) == (D1*D2, D1*D2) || throw(DimensionMismatch())
    B = A[1:D1:D1*D2, 1:D1:D1*D2]
    for i = 2:D1
        B .+= view(A, i:D1:D1*D2, i:D1:D1*D2)
    end
    return B
end
function partialtrace2(A::AbstractMatrix, D1, D2)
    size(A) == (D1*D2, D1*D2) || throw(DimensionMismatch())
    B = A[1:D1, 1:D1]
    for i = 2:D2
        B .+= view(A, 1+(i-1)*D1:i*D1, 1+(i-1)*D1:i*D1)
    end
    return B
end

#
# using LoopVectorization
# function f1(A, B)
#     @tensor C[a,b] := A[x,a,y,b]*B[y,x]
#     return C
# end
# function f2(A,B)
#     D1, D2, D3, D4 = size(A)
#     C = zeros(promote_type(eltype(A), eltype(B)), (D2, D4))
#     @tturbo for i = 1:D1, j = 1:D2, k = 1:D3, l = 1:D4
#         C[j,l] += A[i,j,k,l] * B[k,i]
#     end
#     return C
# end
#
#
# function f3(A::Array{<:Complex, 4}, B::Array{<:Complex,2})
#     D1, D2, D3, D4 = size(A)
#     C = zeros(promote_type(eltype(A), eltype(B)), (D2, D4))
#     Ar = reinterpret(reshape, real(eltype(A)), A)
#     Br = reinterpret(reshape, real(eltype(B)), B)
#     Cr = reinterpret(reshape, real(eltype(C)), C)
#     @tturbo for i = 1:D1, j = 1:D2, k = 1:D3, l = 1:D4
#         Cr[1, j,l] += Ar[1,i,j,k,l] * Br[1,k,i] - Ar[2,i,j,k,l] * Br[2,k,i]
#         Cr[2, j,l] += Ar[2,i,j,k,l] * Br[1,k,i] + Ar[1,i,j,k,l] * Br[2,k,i]
#     end
#     return C
# end
