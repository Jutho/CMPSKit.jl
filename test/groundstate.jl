using Revise
using CMPSKit
using KrylovKit
using OptimKit
using LinearAlgebra
using JLD2

D = 16

Q = Constant(randn(D,D))
R = Constant(randn(D,D))
Ψ = InfiniteCMPS(Q, R)

alg = ConjugateGradient(; verbosity = 2, maxiter = 10^6, gradtol = 1e-4);

H = ∫(∂ψ'*∂ψ - 10000 * ψ'*ψ + 1000 * (ψ')^2*ψ^2, (-Inf,+Inf))


H = ∫(∂ψ'*∂ψ - 1 * ψ'*ψ + 1 * (ψ')^2*ψ^2, (-Inf,+Inf))

# function finalize!(x, f, g, numiter)
#     if mod(numiter, 1000) == 0
#         @save "tempoptresults2.jld" x
#     end
#     return x, f, g
# end
# groundstate(H, Ψ; optalg = alg, linalg = GMRES(; tol = 1e-4), finalize! = finalize!)
groundstate(H, Ψ; optalg = alg, linalg = GMRES(; tol = 1e-6))
