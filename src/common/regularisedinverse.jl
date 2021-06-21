"""
 Take the L2 Tikhonov regularised inverse of a matrix `m`.

 The regularisation parameter is the larger of `delta` (the optional argument that defaults
 to zero) and square root of machine epsilon. The inverse is done using an SVD.
 """
 function reginv(a::AbstractMatrix, δ = defaulttol(a))
     U, S, V = svd(a)
     Sinv = inv.(hypot.(S, δ))
     return  V * Diagonal(Sinv) * U'
 end

 function posreginv(a::AbstractMatrix, δ = defaulttol(a))
     # assumes a is positive definite (or close to)
     D, V = eigen(Hermitian(a))
     Dinv = inv.(hypot.(D, δ))
     return  V * Diagonal(Dinv) * V'
 end

 # computes solution C of A * C + C * A = B for A hermitian
 function symm_sylvester_reg(A::AbstractMatrix, B::AbstractMatrix, δ = zero(eltype(A)))
     n = LinearAlgebra.checksquare(A)
     Λ, U = eigen(Hermitian(A))
     UdCU = U' * B * U
     for j = 1:n
         for i = 1:n
             UdCU[i,j] /= sqrt((Λ[i] + Λ[j])^2 + δ^2)
         end
     end
     C = U*UdCU*U'
     return C
 end
