defaulttol(a) = eps(real(float(one(eltype(a)))))^(1/2)

"""
 Take the L2 Tikhonov regularised inverse of a matrix `m`.

 The regularisation parameter is the larger of `delta` (the optional argument that defaults
 to zero) and square root of machine epsilon. The inverse is done using an SVD.
 """
 function reginv(a::AbstractMatrix, delta = defaulttol(a))
     U, S, V = svd(m)
     Sinv = inv.(hypot.(S, δ))
     return  V * Diagonal(Sinv) * U'
 end

 function posreginv(a::AbstractMatrix, delta = defaulttol(a))
     # assumes a is positive definite (or close to)
     D, V = eigen(Hermitian(A))
     Dinv = inv.(hypot.(D, δ))
     return  V * Diagonal(Dinv) * V'
 end
